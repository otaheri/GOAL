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
import smplx
import numpy as np

from torch import nn, optim

from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.lines import Lines

from psbody.mesh.colors import name_to_rgb

from tools.utils import makepath, makelogger, to_cpu, to_np, to_tensor, create_video

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate

from bps_torch.bps import bps_torch
from tools.utils import aa2rotmat, rotmat2aa, d62rotmat
from models.model_utils import full2bone, full2bone_aa, parms_6D2full

from tools.utils import loc2vel
from tools.utils import smplx_loc2glob


class motion_module(nn.Module):

    def __init__(self,
                 init_params,
                 sbj_model,
                 obj_model,
                 cfg
                 ):
        super(motion_module, self).__init__()

        self.init_params = init_params
        self.sbj_m = sbj_model
        self.obj_m = obj_model
        self.cfg = cfg

        self.past = self.cfg.network.previous_frames
        self.past = 10
        self.future = self.cfg.network.n_out_frames
        self.future = 10

        self.current_params = self.clone_params(init_params)
        self.previous_params = self.clone_params(init_params)

        self.R = torch.eye(3).reshape(1,3,3)
        self.T = torch.zeros(1,3)

        self.device = init_params['transl'].device
        indices = torch.tensor([10,21]).to(self.device)

        self.sbj_params = {'transl':torch.index_select(init_params['transl'][0], dim=0, index = indices),
                           'fullpose':torch.index_select(init_params['fullpose'][0], dim=0, index = indices),
                           'fullpose_rotmat':torch.index_select(init_params['fullpose_rotmat'][0], dim=0, index = indices)}

        indices = torch.tensor([21, 21]).to(self.device)
        self.obj_params = {'transl':torch.index_select(init_params['transl_obj'][0], dim=0, index = indices),
                           'global_orient':torch.index_select(init_params['global_orient_obj'][0], dim=0, index = indices),
                           'global_orient_rotmat':torch.index_select(init_params['global_orient_rotmat_obj'][0], dim=0, index = indices)}



        self.sbj_params, self.obj_params = self.extract_rel_params(self.sbj_params, self.obj_params, repeat=2)

        self.betas = init_params['betas']

        self.num_iters = 0  # number of iterations to generate a motion
        self.n_frames = 2

        self.r_offset = smplx.lbs.vertices2joints(self.sbj_m.J_regressor, self.sbj_m.v_template[0].view(1, -1, 3))[:, 0]
        self.vertex_label_contact = to_tensor(np.load(self.cfg.datasets.vertex_label_contact), dtype=torch.int8).reshape(1, -1)

        self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long)
        self.rhand_idx = to_tensor(np.load(self.cfg.datasets.rh2smplx_ids), dtype=torch.long)
        self.rh_ids_sampled = torch.tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long)

        self.bps_torch = bps_torch()
        # self.mvs = MeshViewers()

        ax_v = np.array([[0., 0., 0.],
                         [1.0, 0., 0.],
                         [0., 1., 0.],
                         [0., 0., 1.]])
        ax_e = [(0, 1), (0, 2), (0, 3)]

        self.axis_l = Lines(ax_v, ax_e, vc=np.eye(4)[:, 1:])

        g_points = np.array([[-.2, 0.0, -.2],
                             [.2, 0.0, .2],
                             [.2, 0.0, -0.2],
                             [-.2, 0.0, .2]])
        g_faces = np.array([[0, 1, 2], [0, 1, 3]])
        self.grnd_mesh = Mesh(v=5 * g_points, f=g_faces, vc=name_to_rgb['gray'])

        cage_points = np.array([[-.2, .0, -.2],
                                [.2, .2, .2],
                                [.2, 0., 0.2],
                                [-.2, .2, -.2]])
        self.cage = [Mesh(v=7 * cage_points, f=[], vc=name_to_rgb['black'])]


    def append_params(self,new_sbj_params, new_obj_params, n_frames = None):
        if n_frames is None:
            self.sbj_params = {k:torch.cat([v[:-1],new_sbj_params[k], v[-1:]], dim=0) for k,v in self.sbj_params.items()}
            self.obj_params = {k:torch.cat([v[:-1],new_obj_params[k], v[-1:]], dim=0) for k,v in self.obj_params.items()}
        else:
            self.sbj_params = {k:torch.cat([v[:-1],new_sbj_params[k][:n_frames], v[-1:]], dim=0) for k,v in self.sbj_params.items()}
            self.obj_params = {k:torch.cat([v[:-1],new_obj_params[k][:n_frames], v[-1:]], dim=0) for k,v in self.obj_params.items()}
        self.n_frames = self.sbj_params['transl'].shape[0]

    def extract_rel_params(self, gen_sbj_params, gen_obj_params=None, repeat=10):

        if gen_obj_params is None:
            gen_obj_params = {k:torch.repeat_interleave(v[-2:-1].clone(), repeat,dim=0) for k,v in self.obj_params.items()}

        sbj_keys = ['transl','fullpose_rotmat']
        obj_keys = ['transl','global_orient','global_orient_rotmat']

        sbj_params_rel = {k:gen_sbj_params[k] for k in sbj_keys if k in gen_sbj_params}
        obj_params_rel = {k:gen_obj_params[k] for k in obj_keys if k in gen_obj_params}

        body_params = parms_6D2full(sbj_params_rel['fullpose_rotmat'],sbj_params_rel['transl'], d62rot=False)
        sbj_params_rel.update(body_params)

        return sbj_params_rel, obj_params_rel

    def rel2glob_params(self, sbj_params_rel, obj_params_rel):

        sbj_params_glob = {k:v.clone() for k,v in sbj_params_rel.items()}
        obj_params_glob = {k:v.clone() for k,v in obj_params_rel.items()}

        r_sbj = sbj_params_rel['fullpose_rotmat'][:, 0]
        t_sbj = sbj_params_rel['transl']
        dev = t_sbj.device

        t_sbj = rotate(t_sbj + self.T.to(dev) + self.r_offset.to(dev), self.R.to(dev)) - self.r_offset.to(dev)

        r_sbj = rotmul(self.R.to(dev), r_sbj)

        sbj_params_glob['transl'] = t_sbj.clone()
        sbj_params_glob['global_orient'] = r_sbj.clone().unsqueeze(1)
        sbj_params_glob['fullpose_rotmat'][:, 0] = r_sbj.clone()

        return sbj_params_glob, obj_params_glob

    def clone_params(self,old_data):
        return {k:v.clone() for k,v in old_data.items()}

    def get_current_params(self):
        return self.clone_params(self.current_params)

    def prepare_data(self):

        T = self.n_frames - 1 # to ignore the last frame
        # duplicate first and last frames to have past and furture frames for them as well
        frames = torch.arange(T).to(torch.long)
        frames = torch.cat([torch.zeros(self.past), frames]).to(torch.long)
        ch = frames[T-1:]
        ch = torch.cat([ch, torch.tensor([-1])]).to(torch.long) # to add the last frame

        sbj_params = {k: v[ch].clone() for k, v in self.sbj_params.items()}
        obj_params = {k: v[ch].clone() for k, v in self.obj_params.items()}
        ############### process the data for the next frame
        R = sbj_params['global_orient'][-2]
        ############# make relative

        R_inv = self.rel_rot(R.clone())
        RR = R_inv.transpose(1,2)

        wind = self.past + 1 + 1 # for the last frame
        motion_sbj, motion_obj, rel_trans = self.glob2rel(sbj_params, obj_params, R_inv, self.r_offset, wind=wind)

        self.current_params['rel_rot'] = R_inv.reshape(1, 3, 3)
        self.current_params['rel_trans'] = rel_trans.reshape(1, 3)

        sbj_output = self.sbj_m(**motion_sbj)
        verts_sbj = sbj_output.vertices
        joints_sbj = sbj_output.joints

        obj_out = self.obj_m(**motion_obj)
        verts_obj = obj_out.vertices

        self.previous_params = self.clone_params(self.current_params)

        ## preparing final data
        sbj_in = {k: v.reshape([1, wind] + list(v.shape[1:])) for k, v in motion_sbj.items()}
        obj_in = {k + '_obj': v.reshape([1, wind] + list(v.shape[1:])) for k, v in motion_obj.items()}

        self.current_params.update(sbj_in)
        self.current_params.update(obj_in)

        self.current_params['joints'] = joints_sbj.reshape(1, wind, -1, 3)
        self.current_params['verts'] = verts_sbj[:,self.verts_ids].reshape(1, wind, -1, 3)

        verts2obj = self.bps_torch.encode(x=verts_obj,
                                          feature_type=['deltas'],
                                          custom_basis=verts_sbj[:, self.verts_ids])['deltas']

        self.current_params['verts2obj'] = verts2obj.reshape(1, wind, -1)

        s_verts_sbj_vel = loc2vel(verts_sbj[:, self.verts_ids], fps=30)
        verts_sbj_vel = loc2vel(verts_sbj, fps=30)
        #
        self.current_params['velocity'] = s_verts_sbj_vel.unsqueeze(0)
        ################
        obj_bps = self.bps['obj'].to(R.device) + motion_obj['transl'][-1:]

        bps_obj = self.bps_torch.encode(x=verts_obj[-1:],
                                        feature_type=['deltas'],
                                        custom_basis=obj_bps)['deltas']

        self.current_params['bps_obj_glob'] = bps_obj.reshape(1, 1, -1)

        bps_rh = self.bps_torch.encode(x=verts_sbj[-1:, self.rhand_idx],
                                       feature_type=['deltas'],
                                       custom_basis=obj_bps)['deltas']

        self.current_params['bps_rh_glob'] = bps_rh.reshape(1, 1, -1)

        ### update global rotation and translation

        self.R = RR.clone()
        self.T = rel_trans.clone()


    def rel_rot(self, R):

        R_smpl_rel = R
        ### for z (forward direction of smpl): keeping the vertical smpl axis always aligned with gravity
        # projecting z on x-z plane
        z_xz = R_smpl_rel[:, :, 2]                                               # forward direction of smpl mesh
        z_xz[:, 1] = 0.                                                          # make the vertical coordinate 0
        z_xz = z_xz / z_xz.norm(dim=-1, keepdim=True)                            # normalize the new vector

        z = torch.tensor([0., 0., 1.]).reshape(list(z_xz.shape)).to(R.device)               # the z axis

        theta = torch.acos(torch.einsum('ij,ij->i', z, z_xz)).reshape(-1, 1)    # find the angle between them
        axis = torch.cross(z, z_xz)                                             # find the rotation axis

        axis_n = axis.norm(dim=-1, keepdim=True)
        if axis_n == 0:
            return torch.eye(3).reshape(1,3,3)

        axis = axis / axis_n
        R_aa = axis * theta
        R_rotmat = aa2rotmat(R_aa)[0]

        # R_rel = torch.matmul(R_vicon2smpl, R_rotmat)
        R_rel = R_rotmat

        return R_rel.transpose(1,2)

    def glob2rel(self, motion_sbj, motion_obj, R, root_offset, wind, rel_trans=None):

        fpose_sbj_rotmat = motion_sbj['fullpose_rotmat']
        global_orient_sbj_rel = rotmul(R, fpose_sbj_rotmat[:, 0])
        fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

        trans_sbj_rel = rotate((motion_sbj['transl'] + root_offset), R) - root_offset
        trans_obj_rel = rotate(motion_obj['transl'], R)

        global_orient_obj_rotmat = aa2rotmat(motion_obj['global_orient'])
        global_orient_obj_rel = rotmul(global_orient_obj_rotmat, R.transpose(1, 2))

        if rel_trans is None:
            rel_trans = trans_sbj_rel.clone().reshape(wind, -1)[-2:-1]
            rel_trans[:, 1] -= rel_trans[:, 1]

        motion_sbj['transl'] = to_tensor(trans_sbj_rel) - rel_trans
        motion_sbj['global_orient'] = to_tensor(global_orient_sbj_rel)
        motion_sbj['fullpose_rotmat'] = fpose_sbj_rotmat

        motion_obj['transl'] = to_tensor(trans_obj_rel) - rel_trans
        motion_obj['global_orient'] = rotmat2aa(to_tensor(global_orient_obj_rel).squeeze()).squeeze()
        motion_obj['global_orient_rotmat'] = to_tensor(global_orient_obj_rel)

        return motion_sbj, motion_obj, rel_trans

    def forward(self, net_output, contact = None, obj_params=None):

        sbj_params = net_output['body_params']
        sbj_params_rel, obj_params_rel = self.extract_rel_params(sbj_params, obj_params, repeat=self.cfg.network.n_out_frames)
        sbj_params_glob, obj_params_glob = self.rel2glob_params(sbj_params_rel, obj_params_rel)

        # self.append_params(sbj_params_glob, obj_params_glob, n_frames=5)
        self.append_params(sbj_params_glob, obj_params_glob)

        self.prepare_data()
        self.num_iters +=1