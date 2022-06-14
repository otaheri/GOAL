# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division

from typing import Callable, Iterator, Union, Optional, List

import os.path as osp
from loguru import logger

import numpy as np

import torch
import torch.nn as nn


from .utils import get_reduction_method
from tools.typing import Tensor

__all__ = [
    'MaskedMSELoss',
    'MaskedL1Loss',
    'VertexEdgeLoss',
    'RotationDistance',
]


def build_loss(type='l2',
               reduction='mean',
               **kwargs
               ) -> nn.Module:

    logger.debug(f'Building loss: {type}')
    if type == 'masked-l2':
        return MaskedMSELoss(reduction=reduction, **kwargs)
    elif type == 'masked-l1':
        return MaskedL1Loss(reduction=reduction, **kwargs)
    elif type == 'vertex-edge':
        return VertexEdgeLoss(reduction=reduction, **kwargs)
    elif type == 'l1':
        return nn.L1Loss(reduction=reduction)
    elif type == 'l2':
        return nn.MSELoss(reduction=reduction)
    elif type == 'rotation':
        return RotationDistance(reduction=reduction, **kwargs)
    else:
        raise ValueError(f'Unknown loss type: {type}')


class MaskedL1Loss(nn.Module):
    def __init__(self, reduction='mean', epsilon=1e-08, **kwargs):
        super(MaskedL1Loss, self).__init__()
        self.reduce_str = reduction
        self.reduce = get_reduction_method(reduction)
        self.epsilon = epsilon

    def forward(
            self,
            input: Tensor,
            target: Tensor,
            mask: Optional[Tensor] = None,
            weight: Tensor = None,
    ):
        diff = input - target
        if weight is not None:
            diff = torch.einsum('ijk, ij -> ijk', diff, weight)
        if mask is None:
            return diff.abs().sum() / (diff.shape[0] + self.epsilon)
        else:
            masked_diff = mask * diff.abs()
            return masked_diff.sum() / mask.sum().clamp_(min=self.epsilon)


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction='mean', epsilon=1e-08, **kwargs):
        super(MaskedMSELoss, self).__init__()
        self.reduce_str = reduction
        self.reduce = get_reduction_method(reduction)
        self.epsilon = epsilon

    def forward(self, input, target, mask=None, weight: Tensor = None,):
        diff = input - target

        if mask is None:
            if weight is None:
                return diff.pow(2).sum() / (diff.shape[0] + self.epsilon)
            else:
                return torch.einsum('ijk, ij -> ijk',diff.pow(2), weight).sum() / (diff.shape[0] + self.epsilon)
        else:
            if weight is None:
                return (mask * diff.pow(2)).sum() / mask.sum().clamp_(min=self.epsilon)
            else:
                return (mask * torch.einsum('ijk, ij -> ijk',diff.pow(2), weight)).sum() / mask.sum().clamp_(min=self.epsilon)


class RotationDistance(nn.Module):
    def __init__(self, reduction='mean', epsilon=1e-7,
                 robustifier='none',
                 **kwargs):
        super(RotationDistance, self).__init__()
        self.reduction = get_reduction_method(reduction)
        self.reduction_str = reduction
        self.epsilon = epsilon

    def extra_repr(self) -> str:
        msg = []
        msg.append(f'Reduction: {self.reduction_str}')
        msg.append(f'Epsilon: {self.epsilon}')
        return '\n'.join(msg)

    def forward(self, module_input, target, weights=None):
        tr = torch.einsum(
            'bij,bij->b',
            [module_input.view(-1, 3, 3),
             target.view(-1, 3, 3)])

        theta = (tr - 1) * 0.5
        loss = torch.acos(
            torch.clamp(theta, -1 + self.epsilon, 1 - self.epsilon))
        if weights is not None:
            loss = loss.view(
                module_input.shape[0], -1) * weights.view(
                    module_input.shape[0], -1)
            return loss.sum() / (
                weights.gt(0).to(loss.dtype).sum() + self.epsilon)
        else:
            return loss.sum() / (module_input.shape[0] + self.epsilon)


class VertexEdgeLoss(nn.Module):
    def __init__(self, norm_type='l2',
                 gt_edge_path='',
                 est_edge_path='',
                 robustifier=None,
                 edge_thresh=0.0, epsilon=1e-8, **kwargs):
        super(VertexEdgeLoss, self).__init__()

        assert norm_type in ['l1', 'l2'], 'Norm type must be [l1, l2]'
        self.norm_type = norm_type
        self.epsilon = epsilon

        gt_edge_path = osp.expandvars(gt_edge_path)
        est_edge_path = osp.expandvars(est_edge_path)
        self.has_connections = osp.exists(gt_edge_path) and osp.exists(
            est_edge_path)
        if self.has_connections:
            gt_edges = np.load(gt_edge_path)
            est_edges = np.load(est_edge_path)

            self.register_buffer(
                'gt_connections', torch.tensor(gt_edges, dtype=torch.long))
            self.register_buffer(
                'est_connections', torch.tensor(est_edges, dtype=torch.long))

    def extra_repr(self):
        msg = [
            f'Norm type: {self.norm_type}',
        ]
        if self.has_connections:
            msg.append(
                f'GT Connections shape: {self.gt_connections.shape}'
            )
            msg.append(
                f'Est Connections shape: {self.est_connections.shape}'
            )
        return '\n'.join(msg)

    def compute_edges(
        self,
        points: Tensor,
        connections: Tensor,
        vertex_axis: int = 1,
    ) -> Tensor:
        ''' Computes the edges from the points and connections'''
        start = torch.index_select(
            points, vertex_axis, connections[:, 0])
        end = torch.index_select(points, vertex_axis, connections[:, 1])
        return start - end

    def forward(
        self,
        gt_vertices: Tensor,
        est_vertices: Tensor,
        mask: Optional[Tensor] = None,
        vertex_axis: int = 1,
    ) -> Tensor:
        if not self.has_connections:
            return 0.0

        # Compute the edges for the ground truth keypoints and the model
        # keypoints. Remove the confidence from the ground truth keypoints
        gt_edges = self.compute_edges(
            gt_vertices, vertex_axis=vertex_axis,
            connections=self.gt_connections)
        est_edges = self.compute_edges(
            est_vertices, vertex_axis=vertex_axis,
            connections=self.est_connections)

        raw_edge_diff = (gt_edges - est_edges)

        batch_size = gt_vertices.shape[0]
        if self.norm_type == 'l2':
            diff = raw_edge_diff.pow(2).sum(dim=-1)
        elif self.norm_type == 'l1':
            diff = raw_edge_diff.abs().sum(dim=-1)
        else:
            raise NotImplementedError(
                f'Loss type not implemented: {self.loss_type}')
        if mask is None:
            return diff.sum() / (batch_size + self.epsilon)
        else:
            return (mask * diff).sum() / mask.sum().clamp_(min=self.epsilon)
