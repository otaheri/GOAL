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
import numpy as np
from smplx import SMPLXLayer
import smplx
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from tools.utils import makepath, to_cpu, to_np, to_tensor, create_video
from tools.utils import loc2vel

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate
from tools.vis_tools import points_to_spheres
from models.model_utils import full2bone, full2bone_aa, parms_6D2full

from omegaconf import OmegaConf
from bps_torch.bps import bps_torch
import chamfer_distance as chd

class MNetOpt(nn.Module):

    def __init__(self,
                 sbj_model,
                 obj_model,
                 cfg,
                 verbose = False
                 ):
        super(MNetOpt, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.cfg = cfg
        self.body_model_cfg = cfg.body_model

        self.sbj_m = sbj_model
        self.obj_m = obj_model

        self.n_out_frames = self.cfg.network.n_out_frames


        self.config_optimizers()

        self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long).to(self.device)
        self.rhand_idx = to_tensor(np.load(self.cfg.losses.rh2smplx_idx)).to(torch.long).to(self.device)
        self.rh_ids_sampled = to_tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long).to(self.device)
        self.feet_ids_sampled = to_tensor(np.load(self.cfg.datasets.verts_feet), dtype=torch.long).to(self.device)
        self.verbose = verbose

        self.bps_torch = bps_torch()
        self.ch_dist = chd.ChamferDistance()

        self.stop = False
        self.li_frames = 0


    def config_optimizers(self):
        bs = self.n_out_frames
        self.bs = bs
        device = self.device
        dtype = self.dtype

        self.opt_params = {
            'global_orient'     : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'body_pose'         : torch.randn(bs, 21*3, device=device, dtype=dtype, requires_grad=True),
            'left_hand_pose'    : torch.randn(bs, 15*3, device=device, dtype=dtype, requires_grad=True),
            'right_hand_pose'   : torch.randn(bs, 15*3, device=device, dtype=dtype, requires_grad=True),
            'jaw_pose'          : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'leye_pose'         : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'reye_pose'         : torch.randn(bs, 1* 3, device=device, dtype=dtype, requires_grad=True),
            'transl'            : torch.zeros(bs, 3, device=device, dtype=dtype, requires_grad=True),
        }

        lr = self.cfg.get('smplx_opt_lr', 5e-3)
        # self.opt_s1 = optim.Adam([self.opt_params[k] for k in ['global_orient', 'transl']], lr=lr)
        # self.opt_s2 = optim.Adam([self.opt_params[k] for k in ['global_orient', 'transl', 'body_pose']], lr=lr)
        self.opt_s3 = optim.Adam([self.opt_params[k] for k in [ 'transl', 'body_pose', 'right_hand_pose']], lr=lr)
        
        self.optimizers = [self.opt_s3]

        self.num_iters = [200]

        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')


    def init_params(self, start_params):

        fullpose_aa = rotmat2aa(start_params['fullpose_rotmat']).reshape(self.n_out_frames, -1)

        start_params_aa = full2bone_aa(fullpose_aa, start_params['transl'])

        for k in self.opt_params.keys():
            self.opt_params[k].data = start_params_aa[k].clone()

    def get_smplx_verts(self, batch, output):

        net_params = output['body_params']
        # verts_offsets = output['dist']
        # net_verts = batch['verts'][:,-2] + 0.01*verts_offsets.reshape(self.bs,-1,3)

        obj_params_gt = {'transl': batch['transl_obj'][:,-1:],
                         'global_orient': batch['global_orient_obj'][:,-1:]}

        obj_output = self.obj_m(**obj_params_gt)

        self.obj_verts = obj_output.vertices
        self.sbj_params = net_params
        # self.net_verts = net_verts

        self.init_params(net_params)

        with torch.no_grad():
            sbj_output = self.sbj_m(**net_params)
            v = sbj_output.vertices.reshape(-1, 10475, 3)
            verts_sampled = v[:, self.verts_ids]

        self.net_verts = torch.cat([batch['verts'][0, -3:-1,] , verts_sampled.clone()], dim = 0)

        self.rh2rh_net = (batch['verts'][:, -1, self.rh_ids_sampled] - self.net_verts[:, self.rh_ids_sampled]).clone()

        self.get_weights()

        return v, verts_sampled

    def get_weights(self):

        rh2rh_min = self.rh2rh_net.norm(dim=-1).min(dim=-1)[0]
        is_close = (rh2rh_min < .1)

        if is_close.any():
            idx = is_close.nonzero(as_tuple=True)[0][0]

            idx_opt = idx - 2
            if idx_opt < 0:
                idx_opt = 0
            self.idx_opt = idx_opt

            idx_match = idx
            if idx < 2:
                idx_match = 2
            self.idx_match = idx_match

            rh_net_vel = loc2vel(self.net_verts[:, self.rh_ids_sampled], 1)
            rh_net_vel_m = rh_net_vel.mean(dim=1).norm(dim=-1)

            weights = torch.cumsum(rh_net_vel_m[idx_match:], dim=0)
            idx2last_dist = self.rh2rh_net[idx_match - 1:idx_match].mean(dim=1).norm(dim=-1)

            weights = (weights / idx2last_dist).reshape(-1, 1, 1)
            weights[weights > 1] = 1.

            if rh_net_vel_m.max() < 0.02:
                self.stop = True

                idx2last_dist = self.rh2rh_net[1:2].mean(dim=1).norm(dim=-1)

                delta = (10*rh_net_vel_m[1] - idx2last_dist)/(55*rh_net_vel_m[1])

                rh_net_vel_m = torch.stack([rh_net_vel_m[1] - i*delta*rh_net_vel_m[1] for i in range(1, 11)], dim=0)
                weights = torch.cumsum(rh_net_vel_m, dim=0)

                weights = (weights / idx2last_dist).reshape(-1, 1, 1)
                weights[weights > 1] = 1.

                self.idx_match = 2
                self.idx_opt = 0

            if weights.max() >= 1.:
                weights[-1] = 1.
                self.stop = True

            self.weights = weights

    def remove_lasts(self, opt_output):

        ones = self.weights == 1
        fullpose_rotmat = opt_output.full_pose.detach()

        if ones.any():
            last = ones.nonzero(as_tuple=True)[0][0]
            ids = self.idx_opt + last
            for k,v in self.opt_params.items():
                self.opt_params[k].data[ids:] = torch.repeat_interleave(self.opt_params[k].data[ids:ids+1], ones.sum(), 0)

            fullpose_rotmat[ids:] = torch.repeat_interleave(fullpose_rotmat[ids:ids+1], ones.sum(), 0)

        body_params = {k: aa2rotmat(v.detach()) for k, v in self.opt_params.items() if v != 'transl'}
        body_params['transl'] = self.opt_params['transl'].detach()
        body_params['fullpose_rotmat'] = fullpose_rotmat

        return body_params



    def calc_loss(self, batch, net_output, stage):


        opt_params = {k:aa2rotmat(v) for k,v in self.opt_params.items() if k!='transl'}
        opt_params['transl'] = self.opt_params['transl']

        output = self.sbj_m(**opt_params, return_full_pose = True)
        verts = output.vertices

        # verts_sampled = verts[:,self.verts_ids]
        # # verts_loss_w = 1
        # #
        rh2obj = self.bps_torch.encode(x=torch.repeat_interleave(self.obj_verts, self.bs, dim=0),
                                       feature_type=['deltas'],
                                       custom_basis=verts[:,self.verts_ids[self.rh_ids_sampled]])['deltas']

        rh2obj_last = batch['verts2obj'][:, -1:].reshape(1, -1, 3).repeat(self.bs, 1,1)[:,self.rh_ids_sampled]

        grasp_rh_pose = torch.repeat_interleave(rotmat2aa(batch['fullpose_rotmat'][:,-1:, 40:]).reshape(1, -1), self.n_out_frames,dim=0).reshape(-1)


        rh_verts_opt = verts[:, self.verts_ids[self.rh_ids_sampled]]
        dist2grnd = verts[:, :, 1].min()
        losses = {}

        linear_rh2rh = self.net_verts[self.idx_match-1:self.idx_match, self.rh_ids_sampled] + self.weights * self.rh2rh_net[self.idx_match-1:self.idx_match]

        losses['linear_rh2rh'] = 20*self.LossL1(rh_verts_opt[self.idx_opt:], linear_rh2rh)
        # losses['rh2rh_offset'] = 5 * self.LossL1(rh2rh_opt, rh2rh_net)
        losses['rh_grasp_pose'] =  0.1 * self.LossL2(torch.exp(-5*self.weights)*grasp_rh_pose.reshape(self.bs,-1).detach(),
                                                     torch.exp(-5*self.weights)*self.opt_params['right_hand_pose'].reshape(self.bs,-1))
        losses['rh2obj'] =  5 * self.LossL1((self.weights==1)* rh2obj[self.idx_opt:], (self.weights==1)*rh2obj_last[self.idx_opt:])
        # losses['rh2obj'] =  10 * self.LossL1(rh2obj[idx:] - (1-weights) * self.rh2rh_net[idx:idx + 1], rh2obj_last[idx:])

        pose_w = 20
        body_loss = {k: w*pose_w*self.LossL2(rotmat2aa(self.sbj_params[k]).detach().reshape(-1), self.opt_params[k].reshape(-1)) for k, w in
                     [('global_orient', 1),
                      ('body_pose', .5),
                      ('left_hand_pose', 1),
                      ('right_hand_pose', .02)
                      ]}

        body_loss['transl'] = 100*pose_w*self.LossL1(self.opt_params['transl'],self.sbj_params['transl'].detach())

        losses.update(body_loss)

        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))
        losses['loss_total'] = loss_total

        return losses, verts, output


    def fitting(self, batch, net_output):

        cnet_verts, cnet_s_verts = self.get_smplx_verts(batch, net_output)

        rh2rh_min = self.rh2rh_net.norm(dim=-1).min(dim=-1)[0]
        is_close = (rh2rh_min < .1)

        if not is_close.any():
            opt_results = {}
            body_params = {k:v.detach() for k,v in net_output['body_params'].items()}
            opt_results['body_params'] = body_params
            opt_results['cnet_verts'] = cnet_verts.detach()
            opt_results['opt_verts'] = cnet_verts.detach()
            return opt_results


        for stg, optimizer in enumerate(self.optimizers):
            for itr in range(self.num_iters[stg]):
                optimizer.zero_grad()
                losses, opt_verts, opt_output = self.calc_loss(batch, net_output, stg)
                losses['loss_total'].backward()
                optimizer.step()
                if self.verbose and itr % 50 == 0:
                    print(self.create_loss_message(losses, stg, itr))


        body_params = self.remove_lasts(opt_output)

        opt_results = {}
        opt_results['body_params'] = body_params
        opt_results['cnet_verts'] = cnet_verts
        opt_results['opt_verts'] = opt_verts

        return opt_results

    @staticmethod
    def create_loss_message(loss_dict, stage=0, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Stage:{stage:02d} - Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'