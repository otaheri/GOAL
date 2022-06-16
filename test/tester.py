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

import os
import shutil
import sys
sys.path.append('..')
import json
import numpy as np
import torch
import smplx
from smplx import SMPLXLayer

from datetime import datetime

from tools.objectmodel import ObjectModel

from tools.utils import makepath, makelogger, to_cpu, to_np, to_tensor, create_video
from loguru import logger

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate

from tools.utils import smplx_loc2glob
from bps_torch.bps import bps_torch

from omegaconf import OmegaConf
from models.mlp import mnet_model
from models.cvae import gnet_model

from data.mnet_dataloader import LoadData, build_dataloader

from tools.utils import aa2rotmat, rotmat2aa, d62rotmat
from models.model_utils import full2bone, full2bone_aa, parms_6D2full
from tools.utils import LOGGER_DEFAULT_FORMAT

from train.motion_module import motion_module
from tools.vis_tools import sp_animation, get_ground

cdir = os.path.dirname(sys.argv[0])

class Trainer:

    def __init__(self,cfg_motion, cfg_static):

        
        self.dtype = torch.float32
        cfg = cfg_motion
        self.cfg = cfg

        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)
        logger_path = makepath(os.path.join(cfg.work_dir, 'V00_test.log'), isfile=True)

        logger.add(logger_path,  backtrace=True, diagnose=True)
        logger.add(lambda x:x,
                   level=cfg.logger_level.upper(),
                   colorize=True,
                   format=LOGGER_DEFAULT_FORMAT
                   )
        self.logger = logger.info

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = cfg.num_gpus
        if use_cuda:
            self.logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))

        self.predict_offsets = cfg.get('predict_offsets', False)
        self.logger(f'Predict offsets: {self.predict_offsets}')

        self.use_exp = cfg.get('use_exp', 0)
        self.logger(f'Use exp function on distances: {self.use_exp}')

        self.body_model_cfg = cfg.body_model
        model_path = os.path.join(self.body_model_cfg.get('model_path', 'data/models'), 'smplx')


        self.body_model = SMPLXLayer(
            model_path=model_path,
            gender='neutral',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.female_model = SMPLXLayer(
            model_path=model_path,
            gender='female',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)
        self.male_model = SMPLXLayer(
            model_path=model_path,
            gender='male',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.object_model = ObjectModel().to(self.device)

        # Create the network
        self.n_out_frames = self.cfg.network.n_out_frames
        self.network_motion = mnet_model(**cfg.network.mnet_model).to(self.device)
        self.network_static = gnet_model(**cfg_static.network.gnet_model).to(self.device)

        self.cfg = cfg
        self.network_motion.cfg = cfg

        self.cfg_static = cfg_static
        self.network_static.cfg = cfg_static

        self.network_motion.load_state_dict(torch.load(cfg.best_model, map_location=self.device), strict=False)
        self.logger('Restored motion grasp trained model from %s' % cfg.best_model)

        self.network_static.load_state_dict(torch.load(cfg_static.best_model, map_location=self.device), strict=False)
        self.logger('Restored static grasp trained model from %s' % cfg_static.best_model)

        self.bps_torch = bps_torch()

        loss_cfg = self.cfg.get('losses', {})

        rh_f = torch.from_numpy(np.load(loss_cfg.rh_faces).astype(np.int32)).view(1, -1, 3)
        self.rh_f = rh_f.repeat(self.cfg.datasets.batch_size, 1, 1).to(torch.long)

        self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long)
        self.feet_ids_sampled = to_tensor(np.load(self.cfg.datasets.verts_feet), dtype=torch.long)
        self.rhand_idx = torch.from_numpy(np.load(loss_cfg.rh2smplx_idx))
        self.rh_ids_sampled = torch.tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long)

        self.bps = torch.load(f'{cdir}/../configs/bps.pt')

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def forward(self, x):

        ##############################################

        bs = x['transl'].shape[0]
        pf = self.cfg.network.previous_frames

        dec_x = {}


        dec_x['fullpose'] = x['fullpose_rotmat'][:,11-pf:11,:,:2,:]
        dec_x['transl'] = x['transl'][:,11-pf:11]

        dec_x['betas'] = x['betas']

        # dec_x['transl_obj'] = x['transl_obj'][:,10:11]

        # verts2last = x['verts'][:, 10:11] - x['verts'][:, -1:]
        verts2last = x['verts'][:, 10:11, self.rh_ids_sampled] - x['verts'][:, -1:, self.rh_ids_sampled]


        if self.use_exp == 0 or self.use_exp != -1:
            # dec_x['feet_vel_exp']  = torch.exp(-self.use_exp * x['velocity'][:, 11-pf:11, self.f_ids.to(dev)].norm(dim=-1))
            # dec_x['feet_dist_exp'] = torch.exp(-self.use_exp * x['verts'][:, 11-pf:11, self.f_ids.to(dev), 1])
            dec_x['vel'] = torch.exp(-self.use_exp * x['velocity'][:, 10:11].norm(dim=-1))
            # dec_x['vel'] = torch.exp(-self.use_exp * x['velocity'][:, 10:11])
            # dec_x['verts2obj_exp'] = torch.exp(-self.use_exp * x['verts2obj'][:, 11-pf:11])

            dec_x['verts_to_last_dist'] = torch.exp(-self.use_exp * verts2last.norm(dim=-1))
        else:
            # dec_x['feet_vel'] = x['velocity'][:, 11-pf:11, self.f_ids.to(dev)].norm(dim=-1)
            # dec_x['feet_dist'] = x['verts'][:, 11-pf:11, self.f_ids.to(dev), 1]
            dec_x['vel'] = x['velocity'][:, 10:11].norm(dim=-1)
            # dec_x['vel'] = x['velocity'][:, 10:11]
            # dec_x['verts2obj'] = x['verts2obj'][:, 11-pf:11]

            dec_x['verts_to_last_dist'] = verts2last.norm(dim=-1)

        dec_x['vel'] = x['velocity'][:, 10:11]
        dec_x['verts'] = x['verts'][:, 10:11]
        dec_x['verts_to_rh'] = verts2last

        dec_x['bps_rh'] = x['bps_rh_glob']

        ########### for the last frame
        # verts2last = x['verts'][:, 10:11, self.rh_ids_sampled] - x['verts'][:,-1:, self.rh_ids_sampled]

        # dec_x['verts_last'] = x['verts'][:,-1:]
        # dec_x['pose_last'] = x['fullpose_rotmat'][:,-1:,:,:2,:]
        # dec_x['trans_last'] = x['transl'][:,-1:]
        # dec_x['rh_bps_last'] = x['bps_rh_glob'][:,-1:]

        #####################################################

        dec_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in dec_x.values()], dim=1)

        pose, trans, dist, rh2last = self.network_motion(dec_x)
        # pose, trans, dist = self.network(dec_x)

        if self.predict_offsets:
            pose_rotmat = d62rotmat(pose).reshape(bs, self.n_out_frames, -1, 3, 3)
            pose = torch.matmul(pose_rotmat,x['fullpose_rotmat'][:, 10:11])
            # pose = pose.reshape(bs,self.n_out_frames,-1) + x['fullpose_rotmat'][:, 10:11, :, :2, :].reshape(bs, 1, -1)
            trans = trans + torch.repeat_interleave(x['transl'][:, 10:11], self.n_out_frames, dim=1).reshape(trans.shape)

        pose = pose.reshape(bs*self.n_out_frames, -1)
        trans = trans.reshape(bs*self.n_out_frames, -1)

        d62rot = pose.shape[-1] == 330
        body_params = parms_6D2full(pose, trans, d62rot= d62rot)

        results = {}
        results['body_params'] = body_params
        results['dist'] = dist
        results['rh2last'] = rh2last

        return  results

    def infer(self, x):


        ##############################################

        bs = x['transl_obj'].shape[0]

        dec_x = {}

        dec_x['betas'] = x['betas']

        dec_x['transl_obj'] = x['transl_obj']

        dec_x['bps_obj'] = x['bps_obj_glob'].reshape(1,-1,3).norm(dim=-1)

        #####################################################

        z_enc = torch.distributions.normal.Normal(
            loc=torch.zeros([1, self.cfg_static.network.gnet_model.latentD], requires_grad=False).to(self.device).type(self.dtype),
            scale=torch.ones([1, self.cfg_static.network.gnet_model.latentD], requires_grad=False).to(self.device).type(self.dtype))

        z_enc_s = z_enc.rsample()
        dec_x['z'] = z_enc_s

        dec_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in dec_x.values()], dim=1)

        net_output = self.network_static.decode(dec_x)

        pose, trans = net_output['pose'], net_output['trans']

        rnet_in, cnet_output, m_refnet_params, f_refnet_params = self.prepare_rnet(x, pose, trans)

        results = {}
        results['z_enc'] = {'mean': z_enc.mean, 'std': z_enc.scale}

        cnet_output.update(net_output)
        results['cnet'] = cnet_output
        results['cnet_f'] = f_refnet_params
        results['cnet_m'] = m_refnet_params

        return results

    def prepare_rnet(self, batch, pose, trans):

        d62rot = pose.shape[-1] == 330
        bparams = parms_6D2full(pose, trans, d62rot=d62rot)

        genders = batch['gender']
        males = genders == 1
        females = ~males

        B = batch['transl_obj'].shape[0]
        v_template = batch['sbj_vtemp'].to(self.device)

        FN = sum(females)
        MN = sum(males)

        f_refnet_params = {}
        m_refnet_params = {}
        cnet_output = {}
        refnet_in = {}


        R_rh_glob = smplx_loc2glob(bparams['fullpose_rotmat'])[:, 21]
        rh_bps = rotate(self.bps['rh'].to(self.device), R_rh_glob)

        if FN > 0:

            f_params = {k: v[females] for k, v in bparams.items()}
            f_params['v_template'] = v_template[females]
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices

            cnet_output['f_verts_full'] = f_verts
            cnet_output['f_params'] = f_params

            f_refnet_params['f_verts2obj'] = self.bps_torch.encode(x=batch['verts_obj'][:,-1][females],
                                              feature_type=['deltas'],
                                              custom_basis=f_verts[:, self.verts_ids])['deltas']
            f_refnet_params['f_rh2obj'] = self.bps_torch.encode(x=batch['verts_obj'][:,-1][females],
                                              feature_type=['deltas'],
                                              custom_basis=f_verts[:, self.rhand_idx])['deltas']

            f_rh_bps = rh_bps[females] + f_output.joints[:, 43:44]

            f_refnet_params['f_bps_obj_rh'] = self.bps_torch.encode(x=batch['verts_obj'][:,-1][females],
                                               feature_type=['deltas'],
                                               custom_basis=f_rh_bps)['deltas']

            # f_refnet_params['f_bps_rh_rh'] = self.bps_torch.encode(x=f_verts[:, self.rhand_idx],
            #                                   feature_type=['dists'],
            #                                   custom_basis=f_rh_bps)['dists']

            refnet_in['f_refnet_in'] = torch.cat([f_params['fullpose_rotmat'][:,:,:2,:].reshape(FN, -1).to(self.device), f_params['transl'].reshape(FN, -1).to(self.device)]
                                  + [v.reshape(FN, -1).to(self.device) for v in f_refnet_params.values()], dim=1)

        if MN > 0:

            m_params = {k: v[males] for k, v in bparams.items()}
            m_params['v_template'] = v_template[males]
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices
            cnet_output['m_verts_full'] = m_verts
            cnet_output['m_params'] = m_params

            m_refnet_params['m_verts2obj'] = self.bps_torch.encode(x=batch['verts_obj'][:,-1][males],
                                                feature_type=['deltas'],
                                                custom_basis=m_verts[:, self.verts_ids])['deltas']
            m_refnet_params['m_rh2obj'] = self.bps_torch.encode(x=batch['verts_obj'][:,-1][males],
                                             feature_type=['deltas'],
                                             custom_basis=m_verts[:, self.rhand_idx])['deltas']

            m_rh_bps = rh_bps[males] + m_output.joints[:, 43:44]

            m_refnet_params['m_bps_obj_rh'] = self.bps_torch.encode(x=batch['verts_obj'][:,-1][males],
                                               feature_type=['deltas'],
                                               custom_basis=m_rh_bps)['deltas']

            # refnet_params['m_bps_rh_rh'] = self.bps_torch.encode(x=m_verts[:, self.rhand_idx],
            #                                   feature_type=['dists'],
            #                                   custom_basis=m_rh_bps)['dists']

            refnet_in['m_refnet_in'] = torch.cat([m_params['fullpose_rotmat'][:, :, :2, :].reshape(MN, -1).to(self.device), m_params['transl'].reshape(MN, -1).to(self.device)]
                                    + [v.reshape(MN, -1).to(self.device) for v in m_refnet_params.values()], dim=1)

        refnet_in = torch.cat([v for v in refnet_in.values()], dim=0)

        return refnet_in, cnet_output, m_refnet_params, f_refnet_params

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='mlp', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)



def loc2vel(loc,fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B-1))
    vel = (loc - loc[idxs])/(1/float(fps))
    return vel

def inference():

    cfg_path_motion = f'configs/MNet_Orig.yaml'
    cfg_motion = OmegaConf.load(cfg_path_motion)
    cfg_motion.batch_size = 1

    cfg_path_static = f'configs/GNet_Orig.yaml'
    cfg_static = OmegaConf.load(cfg_path_static)
    cfg_static.batch_size = 1

    tester = Trainer(cfg_motion, cfg_static)


if __name__ == '__main__':

    inference()
