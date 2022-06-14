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
import torch.nn as nn
import torch.optim as optim

from tools.utils import makepath, to_cpu, to_np, to_tensor, create_video

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate
from models.model_utils import full2bone, full2bone_aa, parms_6D2full
from bps_torch.bps import bps_torch
import chamfer_distance as chd


class GNetOptim(nn.Module):

    def __init__(self,
                 sbj_model,
                 obj_model,
                 cfg,
                 verbose = False
                 ):
        super(GNetOptim, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float32
        self.cfg = cfg
        self.body_model_cfg = cfg.body_model

        self.sbj_m = sbj_model
        self.obj_m = obj_model

        self.config_optimizers()

        self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long)
        self.rhand_idx = to_tensor(np.load(self.cfg.losses.rh2smplx_idx), dtype=torch.long)
        self.rhand_tri = to_tensor(np.load(self.cfg.losses.rh_faces).astype(np.int32))
        self.rh_ids_sampled = torch.tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long)
        self.verbose = verbose

        self.bps_torch = bps_torch()
        self.ch_dist = chd.ChamferDistance()

    def config_optimizers(self):
        bs = 1
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
        self.opt_s3 = optim.Adam([self.opt_params[k] for k in ['global_orient', 'transl', 'body_pose', 'right_hand_pose']], lr=lr)
        
        self.optimizers = [self.opt_s3]

        self.num_iters = [200]

        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')


    def init_params(self, start_params):

        fullpose_aa = rotmat2aa(start_params['fullpose_rotmat']).reshape(1, -1)

        start_params_aa = full2bone_aa(fullpose_aa, start_params['transl'])

        for k in self.opt_params.keys():
            self.opt_params[k].data = torch.repeat_interleave(start_params_aa[k], self.bs, dim=0)

    def get_smplx_verts(self, batch, output):

        B = batch['transl_obj'].shape[0]

        if batch['gender']==1:
            net_params = output['cnet']['m_params']
        else:
            net_params = output['cnet']['f_params']

        obj_params_gt = {'transl': batch['transl_obj'],
                         'global_orient': batch['global_orient_obj']}

        obj_output = self.obj_m(**obj_params_gt)

        self.obj_verts = obj_output.vertices
        self.sbj_params = net_params

        self.init_params(net_params)

        with torch.no_grad():
            sbj_output = self.sbj_m(**net_params)
            v = sbj_output.vertices.reshape(-1, 10475, 3)
            verts_sampled = v[:, self.verts_ids]

        return v, verts_sampled



    def calc_loss(self, batch, net_output, stage):


        opt_params = {k:aa2rotmat(v) for k,v in self.opt_params.items() if k!='transl'}
        opt_params['transl'] = self.opt_params['transl']

        output = self.sbj_m(**opt_params, return_full_pose = True)
        verts = output.vertices
        verts_sampled = verts[:,self.verts_ids]

        rh2obj = self.bps_torch.encode(x=self.obj_verts,
                                       feature_type=['deltas'],
                                       custom_basis=verts[:,self.verts_ids[self.rh_ids_sampled]])['deltas']

        rh2obj_net = net_output['cnet']['dist'].reshape(rh2obj.shape).detach()
        rh2obj_w = torch.exp(-5 * rh2obj_net.norm(dim=-1, keepdim=True))

        gaze_net = net_output['cnet']['gaze']
        gaze_opt_vec = verts_sampled[:, 386] - verts_sampled[:, 387]
        gaze_opt = gaze_opt_vec / gaze_opt_vec.norm(dim=-1, keepdim=True)

        losses = {
            "dist_rh2obj": 2*self.LossL1(rh2obj_w*rh2obj,rh2obj_w*rh2obj_net),
            "grnd_contact": (verts[:,:,1].min() < -.02)*torch.pow(verts[:,:,1].min()+.01, 2),
            "gaze": 1 * self.LossL1(gaze_net.detach(), gaze_opt),
            # 'penet': 1  *torch.pow(rh2obj_penet[is_penet], 2).mean()
        }



        body_loss = {k: self.LossL2(rotmat2aa(self.sbj_params[k]).detach().reshape(-1), self.opt_params[k].reshape(-1)) for k in
                     ['global_orient', 'body_pose', 'left_hand_pose']}

        k = 'right_hand_pose'
        body_loss[k] = .3*self.LossL2(rotmat2aa(self.sbj_params[k]).detach().reshape(-1), self.opt_params[k].reshape(-1))
        body_loss['transl'] = self.LossL1(self.opt_params['transl'],self.sbj_params['transl'].detach())

        losses.update(body_loss)

        loss_total = torch.sum(torch.stack([torch.mean(v) for v in losses.values()]))
        losses['loss_total'] = loss_total

        return losses, verts, output

    def get_peneteration(self,source_mesh, target_mesh):

        source_verts = source_mesh.verts_packed()
        source_normals = source_mesh.verts_normals_packed()

        target_verts = target_mesh.verts_packed()
        target_normals = target_mesh.verts_normals_packed()

        src2trgt, trgt2src, src2trgt_idx, trgt2src_idx = self.ch_dist(source_verts.reshape(1,-1,3).to(self.device), target_verts.reshape(1,-1,3).to(self.device))

        source2target_correspond = target_verts[src2trgt_idx.data.view(-1).long()]

        distance_vector = source_verts - source2target_correspond

        in_out = torch.bmm(source_normals.view(-1, 1, 3), distance_vector.view(-1, 3, 1)).view(-1).sign()

        src2trgt_signed = src2trgt * in_out

        return src2trgt_signed


    def fitting(self, batch, net_output):

        cnet_verts, cnet_s_verts = self.get_smplx_verts(batch, net_output)

        for stg, optimizer in enumerate(self.optimizers):
            for itr in range(self.num_iters[stg]):
                optimizer.zero_grad()
                losses, opt_verts, opt_output = self.calc_loss(batch, net_output, stg)
                losses['loss_total'].backward(retain_graph=True)
                optimizer.step()
                if self.verbose and itr % 50 == 0:
                    print(self.create_loss_message(losses, stg, itr))

        opt_results = {k:aa2rotmat(v.detach()) for k,v in self.opt_params.items() if v != 'transl'}
        opt_results['transl'] = self.opt_params['transl'].detach()
        opt_results['fullpose_rotmat'] = opt_output.full_pose.detach()

        opt_results['cnet_verts'] = cnet_verts
        opt_results['opt_verts'] = opt_verts

        return opt_results

    @staticmethod
    def create_loss_message(loss_dict, stage=0, itr=0):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return f'Stage:{stage:02d} - Iter:{itr:04d} - Total Loss: {loss_dict["loss_total"]:02e} | [{ext_msg}]'
