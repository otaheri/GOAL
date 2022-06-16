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
sys.path.append('.')
sys.path.append('..')
import json
import numpy as np
import torch
import smplx
from smplx import SMPLXLayer

from datetime import datetime
from tensorboardX import SummaryWriter
import glob, time

from psbody.mesh import MeshViewers, Mesh

from psbody.mesh.colors import name_to_rgb
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

    def __init__(self,cfg_motion, cfg_static, inference=False):

        
        self.dtype = torch.float32
        cfg = cfg_motion
        self.cfg = cfg
        self.is_inference = inference

        torch.manual_seed(cfg.seed)

        makepath(cfg.work_dir, isfile=False)
        logger_path = makepath(os.path.join(cfg.work_dir, 'V00_GNet_MNet.log'), isfile=True)

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

        self.data_info = {}
        self.load_data(cfg, inference)


        self.body_model_cfg = cfg.body_model

        self.predict_offsets = cfg.get('predict_offsets', False)
        self.logger(f'Predict offsets: {self.predict_offsets}')

        self.use_exp = cfg.get('use_exp', 0)
        self.logger(f'Use exp function on distances: {self.use_exp}')

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

        # Setup the training losses

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
        self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long)
        self.rhand_idx = torch.from_numpy(np.load(loss_cfg.rh2smplx_idx))
        self.rh_ids_sampled = torch.tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long)

    def load_data(self,cfg, inference):

        self.logger('Base dataset_dir is %s' % self.cfg.datasets.dataset_dir)

        ds_name = 'test'
        self.data_info[ds_name] = {}
        ds_test = LoadData(self.cfg.datasets, split_name=ds_name)
        self.data_info[ds_name]['frame_names'] = ds_test.frame_names
        self.data_info[ds_name]['frame_sbjs'] = ds_test.frame_sbjs
        self.data_info[ds_name]['frame_objs'] = ds_test.frame_objs
        self.data_info[ds_name]['chunk_starts'] = np.array([int(fname.split('_')[-1]) for fname in self.data_info[ds_name]['frame_names'][:, 10]]) == 0
        self.data_info['body_vtmp'] = ds_test.sbj_vtemp
        self.data_info['body_betas'] = ds_test.sbj_betas
        self.data_info['obj_verts'] = ds_test.obj_verts
        self.data_info['obj_info'] = ds_test.obj_info
        self.data_info['sbj_info'] = ds_test.sbj_info
        self.ds_test = build_dataloader(ds_test, split='test', cfg=self.cfg.datasets)

        if not inference:

            ds_name = 'train'
            self.data_info[ds_name] = {}
            ds_train = LoadData(self.cfg.datasets, split_name=ds_name)
            self.data_info[ds_name]['frame_names'] = ds_train.frame_names
            self.data_info[ds_name]['frame_sbjs'] = ds_train.frame_sbjs
            self.data_info[ds_name]['frame_objs'] = ds_train.frame_objs
            self.data_info[ds_name]['chunk_starts'] = np.array([int(fname.split('_')[-1]) for fname in self.data_info[ds_name]['frame_names'][:, 10]]) == 0
            self.data_info['body_vtmp'] = ds_train.sbj_vtemp
            self.data_info['body_betas'] = ds_train.sbj_betas
            self.data_info['obj_verts'] = ds_train.obj_verts
            self.ds_train = build_dataloader(ds_train, split=ds_name, cfg=self.cfg.datasets)

            ds_name = 'val'
            self.data_info[ds_name] = {}
            ds_val = LoadData(self.cfg.datasets, split_name=ds_name)
            self.data_info[ds_name]['frame_names'] = ds_val.frame_names
            self.data_info[ds_name]['frame_sbjs'] = ds_val.frame_sbjs
            self.data_info[ds_name]['frame_objs'] = ds_val.frame_objs
            self.data_info[ds_name]['chunk_starts'] = np.array([int(fname.split('_')[-1]) for fname in self.data_info[ds_name]['frame_names'][:, 10]]) == 0
            self.ds_val = build_dataloader(ds_val, split=ds_name, cfg=self.cfg.datasets)

        self.bps = ds_test.bps
        if not inference:
            self.logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f K' %
                        (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def _get_network(self):
        return self.network_motion.module if isinstance(self.network, torch.nn.DataParallel) else self.network

    def save_network(self):
        torch.save(self.network_motion.module.state_dict()
                   if isinstance(self.network, torch.nn.DataParallel)
                   else self.network_motion.state_dict(), self.cfg.best_model)

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

        bs = x['transl'].shape[0]

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

    def inference_generate(self):

        # torch.set_grad_enabled(False)
        self.network_motion.eval()
        self.network_static.eval()
        device = self.device

        ds_name = 'test'
        data = self.ds_test

        base_movie_path = os.path.join(self.cfg.results_base_dir, self.cfg.expr_ID)

        chunk_starts = self.data_info[ds_name]['chunk_starts']

        visualize = True
        save_meshes = False
        num_samples = 1

        if visualize:
            mvs = MeshViewers()
        else:
            mvs = None


        for batch_id, batch in enumerate(data):

            if not chunk_starts[batch_id]:
                continue

            batch = {k: batch[k].to(self.device) for k in batch.keys()}

            sequence_name = 's' + self.data_info[ds_name]['frame_names'][batch['idx'].to(torch.long)][0][:-2].split('/s')[-1].replace('/', '_')

            gender = batch['gender'].data
            if gender == 2:
                sbj_m = self.female_model
            else:
                sbj_m = self.male_model

            sbj_m.v_template = batch['sbj_vtemp'].to(sbj_m.v_template.device)

            ### object model

            obj_name = self.data_info[ds_name]['frame_names'][batch['idx'].to(torch.long)][0].split('/')[-1].split('_')[0]
            obj_path = os.path.join(self.cfg.datasets.grab_path,'tools/object_meshes/contact_meshes', f'{obj_name}.ply')

            obj_mesh = Mesh(filename=obj_path)
            obj_verts = torch.from_numpy(obj_mesh.v)

            obj_m = ObjectModel(v_template=obj_verts).to(device)
            obj_m.faces = obj_mesh.f

            mov_count = 1
            motion_path = os.path.join(base_movie_path, 'static_and_motion_' + str(mov_count), sequence_name + '_motion.html')
            grasp_path = os.path.join(base_movie_path, 'static_and_motion_' + str(mov_count), sequence_name + '_grasp.html')
            motion_meshes_path = os.path.join(base_movie_path, 'static_and_motion_' + str(mov_count), sequence_name + '_motion_meshes')
            static_meshes_path = os.path.join(base_movie_path, 'static_and_motion_' + str(mov_count), sequence_name + '_static_meshes')

            while os.path.exists(motion_path):
                mov_count += 1
                motion_path = os.path.join(base_movie_path, 'static_and_motion_' + str(mov_count), sequence_name + '_motion.html')
                grasp_path = os.path.join(base_movie_path, 'static_and_motion_' + str(mov_count), sequence_name + '_grasp.html')
                motion_meshes_path = os.path.join(base_movie_path, 'static_and_motion_' + str(mov_count), sequence_name + '_motion_meshes')
                static_meshes_path = os.path.join(base_movie_path, 'static_and_motion_' + str(mov_count), sequence_name + '_static_meshes')

            if save_meshes:
                makepath(motion_meshes_path)
                makepath(static_meshes_path)

            from tools.gnet_optim import GNetOptim as FitSmplxStatic

            fit_smplx_static = FitSmplxStatic(sbj_model=sbj_m,
                                 obj_model=obj_m,
                                 cfg=self.cfg,
                                 verbose=True)

            grnd_mesh, cage, axis_l = get_ground()
            sp_anim_static = sp_animation()
            
            static_grasp_results = []

            batch_static= {}

            for k,v in batch.items():
                if v.ndim>1:
                    if v.shape[1] == 22:
                        batch_static[k] = v.clone()[:,-1]
                    else:
                        batch_static[k] = v.clone()
                else:
                    batch_static[k] = v.clone()

            rel_transl = batch_static['transl_obj'].clone()
            rel_transl[:, 1] -= rel_transl[:, 1]

            batch_static['transl_obj'] -= rel_transl

            for i in range(num_samples):
                print(f'{sequence_name} -- {i}/{num_samples - 1} frames')
                net_output = self.infer(batch_static)

                optim_output = fit_smplx_static.fitting(batch_static, net_output)
                
                static_grasp_results.append(optim_output)

                sbj_cnet = Mesh(v=to_cpu(optim_output['cnet_verts'][0]), f=sbj_m.faces, vc=name_to_rgb['pink'])
                sbj_opt = Mesh(v=to_cpu(optim_output['opt_verts'][0]), f=sbj_m.faces, vc=name_to_rgb['green'])
                obj_i = Mesh(to_cpu(fit_smplx_static.obj_verts[0]), f=obj_mesh.f, vc=name_to_rgb['yellow'])

                if visualize:
                    mvs[0][0].set_static_meshes([sbj_cnet, sbj_opt, obj_i])
                    time.sleep(1)

                if save_meshes:
                    sbj_cnet.write_ply(static_meshes_path+f'/{i:04d}_sbj_coarse.ply')
                    sbj_opt.write_ply(static_meshes_path+f'/{i:04d}_sbj_refine.ply')
                    obj_i.write_ply(static_meshes_path+f'/{i:04d}_obj.ply')

                sp_anim_static.add_frame([sbj_cnet, sbj_opt, obj_i, grnd_mesh], ['coarse_grasp', 'refined_grasp', 'object', 'ground_mesh'])
            ############################
            
            ### Take one of the samples and update the batch based on it #####

            final_grasp = static_grasp_results[0]

            # Transformation from vicon to smplx coordinate frame
            R_v2s = torch.tensor(
                [[1., 0., 0.],
                 [0., 0., -1.],
                 [0., 1., 0.]]).reshape(1, 3, 3).to(self.device)

            rel_rot = batch['rel_rot']

            R_rot = torch.matmul(rel_rot, R_v2s)
            root_offset = smplx.lbs.vertices2joints(sbj_m.J_regressor, sbj_m.v_template[0].view(1, -1, 3))[:, 0]

            fpose_sbj_rotmat = final_grasp['fullpose_rotmat'].clone()
            global_orient_sbj_rel = rotmul(R_rot, fpose_sbj_rotmat[:, 0])
            fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

            trans_sbj_rel = rotate((final_grasp['transl'] + root_offset), R_rot) - root_offset + rel_transl

            batch['transl'][:,-1] = trans_sbj_rel
            batch['fullpose'][:,-1] = rotmat2aa(fpose_sbj_rotmat).reshape(1,-1)
            batch['fullpose_rotmat'][:,-1] = fpose_sbj_rotmat

            grasp_sbj_params = parms_6D2full(fpose_sbj_rotmat, trans_sbj_rel, d62rot=False)

            grasp_sbj_output = sbj_m(**grasp_sbj_params)
            grasp_verts_sampled = grasp_sbj_output.vertices[:, self.verts_ids]

            grasp_obj_params = {'transl': batch['transl_obj'][:,-1],
                             'global_orient': batch['global_orient_obj'][:,-1]}

            grasp_obj_output = obj_m(**grasp_obj_params)

            # grasp_verts2obj = self.bps_torch.encode(x=grasp_obj_output.vertices,
            #                                        feature_type=['deltas'],
            #                                        custom_basis=grasp_verts_sampled)['deltas']

            batch['verts'][:,-1] = grasp_verts_sampled
            # batch['joints'][:,-1] = grasp_sbj_output.joints
            # batch['verts2obj'][:,-1] = grasp_verts2obj.reshape(1,-1)

            ##################################################################

            input_data = {k: batch[k].to(self.device) for k in batch.keys()}

            grasping_motion = motion_module(input_data,
                                     sbj_model=sbj_m,
                                     obj_model=obj_m,
                                     cfg=self.cfg)

            grasping_motion.bps = self.bps
            grasping_motion.mvs = mvs

            input_data = grasping_motion.get_current_params()

            # from tools.verts_to_smplx_motion_grasp_interpolation import FitSmplx
            from tools.mnet_optim import MNetOpt as FitSmplxMotion

            fit_smplx_motion = FitSmplxMotion(sbj_model=sbj_m,
                                             obj_model=obj_m,
                                             cfg=self.cfg)

            fit_smplx_motion.stop = False
            fit_smplx_motion.mvs = mvs

            sp_anim_motion = sp_animation()

            while grasping_motion.num_iters < 10:
                net_output = self.forward(input_data)

                fit_results = fit_smplx_motion.fitting(input_data, net_output)
                grasping_motion(fit_results)
                # grasping_motion(net_output)

                if fit_smplx_motion.stop:
                    break

                input_data = grasping_motion.get_current_params()
                min_dist2obj = input_data['verts2obj'][:, 10].reshape(-1, 3).norm(dim=-1).min()
                min_vertex_offset = net_output['dist'].reshape(-1, 3).norm(dim=-1).max()
                min_dist_offset = net_output['rh2last'].reshape(-1, 3).norm(dim=-1).max()

                if min_dist2obj<.003 and min_dist_offset<.2:
                    break


            sbj_params = {k: v.clone() for k, v in grasping_motion.sbj_params.items()}
            obj_params = {k: v.clone() for k, v in grasping_motion.obj_params.items()}

            sbj_output_glob = sbj_m(**sbj_params)
            verts_sbj_glob = sbj_output_glob.vertices
            joints_sbj_glob = sbj_output_glob.joints

            obj_out_glob = obj_m(**obj_params)
            verts_obj_glob = obj_out_glob.vertices

            grnd_mesh, cage, axis_l = get_ground()
            #################
            for i in range(grasping_motion.n_frames - 1):
                sbj_i = Mesh(v=to_cpu(verts_sbj_glob[i]), f=sbj_m.faces, vc=name_to_rgb['pink'])
                obj_i = Mesh(v=to_cpu(verts_obj_glob[i]), f=obj_mesh.f, vc=name_to_rgb['yellow'])

                if visualize:
                    mvs[0][0].set_static_meshes([sbj_i, obj_i, grnd_mesh])
                    mvs[0][0].set_static_lines([grasping_motion.axis_l])

                if save_meshes:
                    sbj_i.write_ply(motion_meshes_path + f'/{i:04d}_sbj.ply')
                    obj_i.write_ply(motion_meshes_path + f'/{i:04d}_obj.ply')

                sp_anim_motion.add_frame([sbj_i, obj_i, grnd_mesh], ['sbj_mesh', 'obj_mesh', 'ground_mesh'])
                ############################
            if save_meshes:
                sp_anim_motion.save_animation(motion_path)
                sp_anim_static.save_animation(grasp_path)
            ############################

def loc2vel(loc,fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B-1))
    vel = (loc - loc[idxs])/(1/float(fps))
    return vel

def inference():

    import argparse

    parser = argparse.ArgumentParser(description='GOAL-Testing')

    parser.add_argument('--work-dir',
                        required=True,
                        type=str,
                        help='The path to the folder to save results')

    parser.add_argument('--grab-path',
                        required=True,
                        type=str,
                        help='The path to the folder that contains GRAB data')

    parser.add_argument('--smplx-path',
                        required = True,
                        type=str,
                        help='The path to the folder containing SMPLX models')


    cmd_args = parser.parse_args()

    # gnet_cfg_path = args.gnet_cfg_path
    # mnet_cfg_path = args.mnet_cfg_path
    # obj_path = args.obj_path
    # smplx_path = args.smplx_path


    best_gnet = f'{cdir}/../models/GNet_model.pt'
    best_mnet = f'{cdir}/../models/MNet_model.pt'


    cfg_path_motion = f'{cdir}/../configs/MNet_orig.yaml'
    cfg_motion = OmegaConf.load(cfg_path_motion)
    cfg_motion.batch_size = 1
    cfg_motion.best_model = best_mnet

    cfg_motion.output_folder = cmd_args.work_dir
    cfg_motion.results_base_dir = os.path.join(cfg_motion.output_folder, 'results')
    cfg_motion.work_dir = os.path.join(cfg_motion.output_folder, 'GOAL_test')

    cfg_motion.datasets.dataset_dir = os.path.join(cmd_args.grab_path,'MNet_data')
    cfg_motion.datasets.grab_path = cmd_args.grab_path


    cfg_path_static = f'{cdir}/../configs/GNet_orig.yaml'
    cfg_static = OmegaConf.load(cfg_path_static)
    cfg_static.batch_size = 1
    cfg_static.best_model = best_gnet

    cfg_static.output_folder = cmd_args.work_dir
    cfg_static.results_base_dir = os.path.join(cfg_static.output_folder, 'results')
    cfg_static.work_dir = os.path.join(cfg_static.output_folder, 'GOAL_test')

    cfg_static.datasets.dataset_dir = os.path.join(cmd_args.grab_path,'GNet_data')
    cfg_static.datasets.grab_path = cmd_args.grab_path

    cfg_motion.body_model.model_path = cfg_static.body_model.model_path = cmd_args.smplx_path

    tester = Trainer(cfg_motion, cfg_static, inference=True)

    tester.inference_generate()


if __name__ == '__main__':

    inference()
