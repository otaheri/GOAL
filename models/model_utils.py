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

import torch
import torch.nn as nn
import torch.nn.init as nninit

from loguru import logger

from tools.utils import aa2rotmat, rotmat2aa, d62rotmat


@torch.no_grad()
def init_weights(
    layer,
    name='',
    init_type='xavier',
    distr='uniform',
    gain=1.0,
    activ_type='leaky-relu', lrelu_slope=0.01, **kwargs
):
    if len(name) < 1:
        name = str(layer)
    logger.info(
        f'Initializing {name} with {init_type}_{distr}: gain={gain}')
    weights = layer.weight
    if init_type == 'xavier':
        if distr == 'uniform':
            nninit.xavier_uniform_(weights, gain=gain)
        elif distr == 'normal':
            nninit.xavier_normal_(weights, gain=gain)
        else:
            raise ValueError(
                f'Unknown distribution "{distr}" for Kaiming init')
    elif init_type == 'kaiming':
        activ_type = activ_type.replace('-', '_')
        if distr == 'uniform':
            nninit.kaiming_uniform_(weights, a=lrelu_slope,
                                    nonlinearity=activ_type)
        elif distr == 'normal':
            nninit.kaiming_normal_(weights, a=lrelu_slope,
                                   nonlinearity=activ_type)
        else:
            raise ValueError(
                f'Unknown distribution "{distr}" for Kaiming init')


def parms_decode_full(pose,trans):

    bs = trans.shape[0]

    pose_full = d62rotmat(pose)
    pose = pose_full.reshape([bs, 1, -1, 9])
    pose = rotmat2aa(pose).reshape(bs, -1)

    body_parms = full2bone_aa(pose,trans)
    pose_full = pose_full.reshape([bs, -1, 3, 3])
    body_parms['fullpose_rotmat'] = pose_full
    body_parms['fullpose'] = pose

    return body_parms

def full2bone_aa(pose,trans):

    bs = trans.shape[0]
    if pose.ndim>2:
        pose = pose.reshape([bs, 1, -1, 9])
        pose = rotmat2aa(pose).reshape(bs, -1)

    global_orient = pose[:, :3]
    body_pose = pose[:, 3:66]
    jaw_pose  = pose[:, 66:69]
    leye_pose = pose[:, 69:72]
    reye_pose = pose[:, 72:75]
    left_hand_pose = pose[:, 75:120]
    right_hand_pose = pose[:, 120:]

    body_parms = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                  'transl': trans}
    return body_parms

def full2bone(pose,trans):

    global_orient = pose[:, 0:1]
    body_pose = pose[:, 1:22]
    jaw_pose  = pose[:, 22:23]
    leye_pose = pose[:, 23:24]
    reye_pose = pose[:, 24:25]
    left_hand_pose = pose[:, 25:40]
    right_hand_pose = pose[:, 40:]

    body_parms = {'global_orient': global_orient, 'body_pose': body_pose,
                  'jaw_pose': jaw_pose, 'leye_pose': leye_pose, 'reye_pose': reye_pose,
                  'left_hand_pose': left_hand_pose, 'right_hand_pose': right_hand_pose,
                  'transl': trans}
    return body_parms

def parms_6D2full(pose,trans, d62rot=True):

    bs = trans.shape[0]

    if d62rot:
        pose = d62rotmat(pose)
    pose = pose.reshape([bs, -1, 3, 3])

    body_parms = full2bone(pose,trans)
    body_parms['fullpose_rotmat'] = pose

    return body_parms

