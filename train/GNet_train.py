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

from smplx import SMPLXLayer


from datetime import datetime
from tools.train_tools import EarlyStopping


from torch import nn, optim

from pytorch3d.structures import Meshes
from tensorboardX import SummaryWriter

import glob, time

from psbody.mesh import MeshViewers, Mesh
from psbody.mesh.lines import Lines

from psbody.mesh.colors import name_to_rgb
from tools.objectmodel import ObjectModel

from tools.utils import makepath, makelogger, to_cpu, to_np, to_tensor, create_video
from loguru import logger

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate

from tools.utils import smplx_loc2glob

from bps_torch.bps import bps_torch


from omegaconf import OmegaConf

from models.cvae import gnet_model
from losses import build_loss
from optimizers import build_optimizer
from data.gnet_dataloader import LoadData, build_dataloader

from tools.utils import aa2rotmat, rotmat2aa, d62rotmat
from models.model_utils import full2bone, full2bone_aa, parms_6D2full
from tools.train_tools import v2v
from tqdm import tqdm

from tools.vis_tools import sp_animation, get_ground
from tools.utils import LOGGER_DEFAULT_FORMAT
cdir = os.path.dirname(sys.argv[0])


class Trainer:

    def __init__(self,cfg, inference=False):

        
        self.dtype = torch.float32
        self.cfg = cfg
        self.is_inference = inference

        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)
        logger_path = makepath(os.path.join(cfg.work_dir, '%s_%s.log' % (cfg.expr_ID, 'train' if not inference else 'test')), isfile=True)

        logger.add(logger_path,  backtrace=True, diagnose=True)
        logger.add(lambda x:x,
                   level=cfg.logger_level.upper(),
                   colorize=True,
                   format=LOGGER_DEFAULT_FORMAT
                   )
        self.logger = logger.info

        summary_logdir = os.path.join(cfg.work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        self.logger('[%s] - Started training XXX, experiment code %s' % (cfg.expr_ID, starttime))
        self.logger('tensorboard --logdir=%s' % summary_logdir)
        self.logger('Torch Version: %s\n' % torch.__version__)

        stime = datetime.now().replace(microsecond=0)
        shutil.copy2(sys.argv[0], os.path.join(cfg.work_dir, os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % datetime.strftime(stime, '%Y%m%d_%H%M'))))

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = cfg.num_gpus
        if use_cuda:
            self.logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))

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
        self.network = gnet_model(**cfg.network.gnet_model).to(self.device)

        # Setup the training losses
        self.loss_setup()


        if cfg.num_gpus > 1:
            self.network = nn.DataParallel(self.network)
            self.logger("Training on Multiple GPU's")

        vars_network = [var[1] for var in self.network.named_parameters()]
        n_params = sum(p.numel() for p in vars_network if p.requires_grad)
        self.logger('Total Trainable Parameters for network is %2.2f M.' % ((n_params) * 1e-6))

        self.configure_optimizers()

        self.best_loss = np.inf

        self.epochs_completed = 0
        self.cfg = cfg
        self.network.cfg = cfg

        if inference and cfg.best_model is None:
            cfg.best_model = sorted(glob.glob(os.path.join(cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model.pt')))[-1]
        if cfg.best_model is not None:
            self._get_network().load_state_dict(torch.load(cfg.best_model, map_location=self.device), strict=False)
            self.logger('Restored trained model from %s' % cfg.best_model)

        self.bps_torch = bps_torch()


    def loss_setup(self):

        self.logger('Configuring the losses!')

        loss_cfg = self.cfg.get('losses', {})

        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')
        self.Lossbce = nn.BCELoss(reduction='mean')

        # Edge loss
        edge_loss_cfg = loss_cfg.get('edge', {})
        self.edge_loss = build_loss(**edge_loss_cfg)
        self.edge_loss_weight = edge_loss_cfg.get('weight', 0.0)
        self.logger(f'Edge loss, weight: {self.edge_loss}, {self.edge_loss_weight}')

        # Vertex loss
        # TODO: Add denser vertex sampling
        vertex_loss_cfg = loss_cfg.get('vertices', {})
        self.vertex_loss_weight = vertex_loss_cfg.get('weight', 0.0)
        self.vertex_loss = build_loss(**vertex_loss_cfg)
        self.logger(f'Vertex loss, weight: {self.vertex_loss},'
                    f' {self.vertex_loss_weight}')

        vertex_consist_loss_cfg = loss_cfg.get('vertices_consist', {})
        self.vertex_consist_loss_weight = vertex_consist_loss_cfg.get('weight', 0.0)
        # self.vertex_loss = build_loss(**vertex_loss_cfg)
        self.logger(f'Vertex consist loss weight: {self.vertex_consist_loss_weight}')

        rh_vertex_loss_cfg = loss_cfg.get('rh_vertices', {})
        self.rh_vertex_loss_weight = rh_vertex_loss_cfg.get('weight', 0.0)
        self.rh_vertex_loss = build_loss(**rh_vertex_loss_cfg)
        self.logger(f'Right Hand Vertex loss, weight: {self.rh_vertex_loss},'
                     f' {self.rh_vertex_loss_weight}')

        feet_vertex_loss_cfg = loss_cfg.get('feet_vertices', {})
        self.feet_vertex_loss_weight = feet_vertex_loss_cfg.get('weight', 0.0)
        self.feet_vertex_loss = build_loss(**feet_vertex_loss_cfg)
        self.logger(f'Feet Vertex loss, weight: {self.feet_vertex_loss},'
                     f' {self.feet_vertex_loss_weight}')

        pose_loss_cfg = loss_cfg.get('pose', {})
        self.pose_loss_weight = pose_loss_cfg.get('weight', 0.0)
        self.pose_loss = build_loss(**pose_loss_cfg)
        self.logger(f'Pose loss, weight: {self.pose_loss},'
                    f' {self.pose_loss}')

        velocity_loss_cfg = loss_cfg.get('velocity', {})
        self.velocity_loss_weight = velocity_loss_cfg.get('weight', 0.0)
        self.velocity_loss = build_loss(**velocity_loss_cfg)

        self.logger(f'Velocity loss, weight: {self.velocity_loss},'
                    f' {self.velocity_loss_weight}')

        acceleration_loss_cfg = loss_cfg.get('acceleration', {})
        self.acceleration_loss_weight = acceleration_loss_cfg.get('weight', 0.0)
        self.acceleration_loss = build_loss(**acceleration_loss_cfg)
        self.logger(
            f'Acceleration loss, weight: {self.acceleration_loss},'
            f' {self.acceleration_loss_weight}')

        contact_loss_cfg = loss_cfg.get('contact', {})
        self.contact_loss_weight = contact_loss_cfg.get('weight', 0.0)
        self.logger(
            f'Contact loss, weight: '
            f' {self.contact_loss_weight}')

        kl_loss_cfg = loss_cfg.get('kl_loss', {})
        self.kl_loss_weight = kl_loss_cfg.get('weight', 0.0)
        self.logger(
            f'KL loss, weight: '
            f' {self.kl_loss_weight}')

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
            self.ds_val = build_dataloader(ds_val, split=ds_name, cfg=self.cfg.datasets)

        self.bps = ds_test.bps
        self.bps_torch = bps_torch()
        if not inference:
            self.logger('Dataset Train, Vald, Test size respectively: %.2f M, %.2f K, %.2f K' %
                        (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def _get_network(self):
        return self.network.module if isinstance(self.network, torch.nn.DataParallel) else self.network

    def save_network(self):
        torch.save(self.network.module.state_dict()
                   if isinstance(self.network, torch.nn.DataParallel)
                   else self.network.state_dict(), self.cfg.best_model)

    def forward(self, x):

        if self.is_inference:
            return self.infer(x)
        ##############################################

        bs = x['transl'].shape[0]

        dec_x = {}
        enc_x = {}

        enc_x['fullpose'] = x['fullpose_rotmat'][:,:,:2,:]
        enc_x['transl'] = x['transl']

        dec_x['betas'] = x['betas']
        enc_x['betas'] = x['betas']

        dec_x['transl_obj'] = x['transl_obj']
        enc_x['transl_obj'] = x['transl_obj']


        if self.use_exp != 0 and self.use_exp != -1:
            enc_x['verts2obj_exp'] = torch.exp(-self.use_exp * x['verts2obj'])
        else:
            enc_x['verts2obj'] = x['verts2obj']

        enc_x['verts'] = x['verts']

        dec_x['bps_obj'] = x['bps_obj_glob'].norm(dim=-1)
        enc_x['bps_obj'] = x['bps_obj_glob'].norm(dim=-1)

        #####################################################
        enc_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in enc_x.values()], dim=1)

        z_enc = self.network.encode(enc_x)
        z_enc_s = z_enc.rsample()
        dec_x['z'] = z_enc_s

        dec_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in dec_x.values()], dim=1)

        net_output = self.network.decode(dec_x)

        pose, trans = net_output['pose'], net_output['trans']

        rnet_in, cnet_output, m_refnet_params, f_refnet_params = self.prepare_rnet(x, pose, trans)

        results = {}
        results['z_enc'] = {'mean': z_enc.mean, 'std': z_enc.scale}

        cnet_output.update(net_output)
        results['cnet'] = cnet_output
        results['cnet_f'] = f_refnet_params
        results['cnet_m'] = m_refnet_params

        return  results

    def prepare_rnet(self, batch, pose, trans):

        d62rot = pose.shape[-1] == 330
        bparams = parms_6D2full(pose, trans, d62rot=d62rot)

        genders = batch['gender']
        males = genders == 1
        females = ~males

        B, _ = batch['transl_obj'].shape
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
            self.female_model.v_template = v_template[females].clone()
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices

            cnet_output['f_verts_full'] = f_verts
            cnet_output['f_params'] = f_params

            f_refnet_params['f_verts2obj'] = self.bps_torch.encode(x=batch['verts_obj'][females],
                                              feature_type=['deltas'],
                                              custom_basis=f_verts[:, self.verts_ids])['deltas']
            f_refnet_params['f_rh2obj'] = self.bps_torch.encode(x=batch['verts_obj'][females],
                                              feature_type=['deltas'],
                                              custom_basis=f_verts[:, self.rhand_idx])['deltas']

            f_rh_bps = rh_bps[females] + f_output.joints[:, 43:44]

            f_refnet_params['f_bps_obj_rh'] = self.bps_torch.encode(x=batch['verts_obj'][females],
                                               feature_type=['deltas'],
                                               custom_basis=f_rh_bps)['deltas']

            refnet_in['f_refnet_in'] = torch.cat([f_params['fullpose_rotmat'][:,:,:2,:].reshape(FN, -1).to(self.device), f_params['transl'].reshape(FN, -1).to(self.device)]
                                  + [v.reshape(FN, -1).to(self.device) for v in f_refnet_params.values()], dim=1)

        if MN > 0:

            m_params = {k: v[males] for k, v in bparams.items()}
            self.male_model.v_template = v_template[males].clone()
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices
            cnet_output['m_verts_full'] = m_verts
            cnet_output['m_params'] = m_params

            m_refnet_params['m_verts2obj'] = self.bps_torch.encode(x=batch['verts_obj'][males],
                                                feature_type=['deltas'],
                                                custom_basis=m_verts[:, self.verts_ids])['deltas']
            m_refnet_params['m_rh2obj'] = self.bps_torch.encode(x=batch['verts_obj'][males],
                                             feature_type=['deltas'],
                                             custom_basis=m_verts[:, self.rhand_idx])['deltas']

            m_rh_bps = rh_bps[males] + m_output.joints[:, 43:44]

            m_refnet_params['m_bps_obj_rh'] = self.bps_torch.encode(x=batch['verts_obj'][males],
                                               feature_type=['deltas'],
                                               custom_basis=m_rh_bps)['deltas']

            refnet_in['m_refnet_in'] = torch.cat([m_params['fullpose_rotmat'][:, :, :2, :].reshape(MN, -1).to(self.device), m_params['transl'].reshape(MN, -1).to(self.device)]
                                    + [v.reshape(MN, -1).to(self.device) for v in m_refnet_params.values()], dim=1)

        refnet_in = torch.cat([v for v in refnet_in.values()], dim=0)

        return refnet_in, cnet_output, m_refnet_params, f_refnet_params

    def infer(self, x):
        ##############################################
        bs = x['transl'].shape[0]
        dec_x = {}

        dec_x['betas'] = x['betas']
        dec_x['transl_obj'] = x['transl_obj']
        dec_x['bps_obj'] = x['bps_obj_glob'].norm(dim=-1)
        #####################################################
        z_enc = torch.distributions.normal.Normal(
            loc=torch.zeros([1, self.cfg.network.gnet_model.latentD], requires_grad=False).to(self.device).type(self.dtype),
            scale=torch.ones([1, self.cfg.network.gnet_model.latentD], requires_grad=False).to(self.device).type(self.dtype))

        z_enc_s = z_enc.rsample()
        dec_x['z'] = z_enc_s

        dec_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in dec_x.values()], dim=1)

        net_output = self.network.decode(dec_x)

        pose, trans = net_output['pose'], net_output['trans']

        rnet_in, cnet_output, m_refnet_params, f_refnet_params = self.prepare_rnet(x, pose, trans)

        results = {}
        results['z_enc'] = {'mean': z_enc.mean, 'std': z_enc.scale}

        cnet_output.update(net_output)
        results['cnet'] = cnet_output
        results['cnet_f'] = f_refnet_params
        results['cnet_m'] = m_refnet_params

        return results
        #####################################################

    def train(self):

        self.network.train()
        save_every_it = len(self.ds_train) / self.cfg.summary_steps
        train_loss_dict = {}

        for it, batch in enumerate(self.ds_train):
            batch = {k: batch[k].to(self.device) for k in batch.keys()}

            self.optimizer.zero_grad()
            # torch.autograd.set_detect_anomaly(True)
            
            output = self.forward(batch)

            loss_total, losses_dict = self.get_loss(batch, it, output)

            loss_total.backward()
            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in losses_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                train_msg = self.create_loss_message(cur_train_loss_dict,
                                                    expr_ID=self.cfg.expr_ID,
                                                    epoch_num=self.epochs_completed,
                                                    model_name='cvae_grasp_xyz',
                                                    it=it,
                                                    try_num=0,
                                                    mode='train')

                self.logger(train_msg)

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}

        return train_loss_dict

    def evaluate(self, ds_name='val'):
        self.network.eval()

        eval_loss_dict = {}

        data = self.ds_val if ds_name == 'val' else self.ds_test

        with torch.no_grad():
            for it, batch in enumerate(data):

                batch = {k: batch[k].to(self.device) for k in batch.keys()}

                self.optimizer.zero_grad()
                # torch.autograd.set_detect_anomaly(True)

                output = self.forward(batch)

                loss_total, losses_dict = self.get_loss(batch, it, output)

                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in losses_dict.items()}

            eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}

        return eval_loss_dict

    def get_loss(self, batch, batch_idx, results):

        enc_z = results['z_enc']
        cnet = results['cnet']

        genders = batch['gender']
        males = genders == 1
        females = ~males

        B, _ = batch['transl_obj'].shape
        v_template = batch['sbj_vtemp']

        FN = sum(females)
        MN = sum(males)

        params_gt = parms_6D2full(batch['fullpose_rotmat'],
                                  batch['transl'],
                                  d62rot=False)
        if FN > 0:
            f_params_gt = {k: v[females] for k, v in params_gt.items()}
            self.female_model.v_template = v_template[females].clone()
            f_output_gt = self.female_model(**f_params_gt)
            f_verts_gt = f_output_gt.vertices

        if MN > 0:
            m_params_gt = {k: v[males] for k, v in params_gt.items()}
            self.male_model.v_template = v_template[males].clone()
            m_output_gt = self.male_model(**m_params_gt)
            m_verts_gt = m_output_gt.vertices

        losses = {}

        # vertex loss
        if self.vertex_loss_weight > 0:
            losses['cnet_vertices'] = 0
            if FN > 0:
                losses['cnet_vertices'] += self.vertex_loss(f_verts_gt, cnet['f_verts_full'])
            if MN > 0:
                losses['cnet_vertices'] += self.vertex_loss(m_verts_gt, cnet['m_verts_full'])

            losses['cnet_vertices'] *= self.vertex_loss_weight

        if self.pose_loss_weight > 0:
            losses['cnet_pose'] = 0
            losses['cnet_trans'] = 0
            if FN>0:
                losses['cnet_pose'] += self.LossL2(f_params_gt['fullpose_rotmat'], cnet['f_params']['fullpose_rotmat'])
                losses['cnet_trans'] += self.LossL1(f_params_gt['transl'], cnet['f_params']['transl'])
            if MN>0:
                losses['cnet_pose'] += self.LossL2(m_params_gt['fullpose_rotmat'], cnet['m_params']['fullpose_rotmat'])
                losses['cnet_trans'] += self.LossL1(m_params_gt['transl'], cnet['m_params']['transl'])

            losses['cnet_pose'] *= self.pose_loss_weight
            losses['cnet_trans'] *= self.pose_loss_weight


        # right hand vertex loss
        if self.rh_vertex_loss_weight > 0:
            losses['cnet_rh_vertices'] = 0
            if FN > 0:
                losses['cnet_rh_vertices'] += self.vertex_loss(f_verts_gt[:, self.rhand_idx], cnet['f_verts_full'][:, self.rhand_idx])
            if MN > 0:
                losses['cnet_rh_vertices'] += self.vertex_loss(m_verts_gt[:, self.rhand_idx], cnet['m_verts_full'][:, self.rhand_idx])

            losses['cnet_rh_vertices'] *= self.rh_vertex_loss_weight

        rh2obj_gt = batch['verts2obj'][:, self.rh_ids_sampled]
        rh2obj_w = torch.exp(-5*rh2obj_gt.norm(dim=-1, keepdim=True))
        losses['rh2obj'] = self.vertex_loss_weight*self.LossL1(rh2obj_w*rh2obj_gt, rh2obj_w*cnet['dist'].reshape(rh2obj_gt.shape))

        gaze_vec_gt = batch['verts'][:,386] - batch['verts'][:,387]

        gaze_vec_gt_n = gaze_vec_gt/gaze_vec_gt.norm(dim=-1, keepdim=True)

        losses['gaze'] = 3*self.LossL1(gaze_vec_gt_n, cnet['gaze'])

        e_z = torch.distributions.normal.Normal(enc_z['mean'], enc_z['std'])  # encoder distribution

        n_z = torch.distributions.normal.Normal(
            loc=torch.zeros([self.cfg.datasets.batch_size, self.cfg.network.gnet_model.latentD], requires_grad=False).to(self.device).type(self.dtype),
            scale=torch.ones([self.cfg.datasets.batch_size, self.cfg.network.gnet_model.latentD], requires_grad=False).to(self.device).type(self.dtype))

        losses['loss_kl_encoder']   = self.kl_loss_weight * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(e_z, n_z), dim=[1]))  # kl between the encoder and normal distribution

        with torch.no_grad():
            loss_v2v = []

            if FN > 0:
                loss_v2v.append(v2v(f_verts_gt,
                                     cnet['f_verts_full'],
                                     mean=False)
                                 )
            if MN > 0:
                loss_v2v.append(v2v(m_verts_gt,
                                     cnet['m_verts_full'],
                                     mean=False)
                                 )

            loss_v2v = torch.cat(loss_v2v, dim=0).mean(dim=-1).sum()

        loss_total = torch.stack(list(losses.values())).sum()
        losses['loss_total'] = loss_total
        losses['loss_v2v'] = loss_v2v

        return loss_total, losses

    def set_loss_weights(self):

        if self.epochs_completed > 3:
            self.pose_loss_weight = 2
            self.vertex_loss_weight = 15
        else:
            self.vertex_loss_weight = 1

        if self.epochs_completed > 10:
            self.rh_vertex_loss_weight = 8


    def fit(self, n_epochs=None, message=None):

        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr = np.inf

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            train_loss_dict = self.train()
            eval_loss_dict  = self.evaluate()

            self.set_loss_weights()


            self.lr_scheduler.step(eval_loss_dict['loss_v2v'])
            cur_lr = self.optimizer.param_groups[0]['lr']

            if cur_lr != prev_lr:
                self.logger('--- learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr

            with torch.no_grad():
                eval_msg = Trainer.create_loss_message(eval_loss_dict, expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                        model_name='cvae_grasp_xyz',
                                                        try_num=0, mode='evald')
                if eval_loss_dict['loss_v2v'] < self.best_loss:

                    self.cfg.best_model = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'E%03d_model.pt' % (self.epochs_completed)), isfile=True)
                    self.save_network()
                    self.logger(eval_msg + ' ** ')
                    self.best_loss = eval_loss_dict['loss_v2v']

                else:
                    self.logger(eval_msg)

                self.swriter.add_scalars('total_loss/scalars',
                                         {'train_loss_total': train_loss_dict['loss_total'],
                                         'evald_loss_total': eval_loss_dict['loss_total'], },
                                         self.epochs_completed)

            if self.early_stopping(eval_loss_dict['loss_v2v']):
                self.logger('Early stopping the training!')
                break

            self.epochs_completed += 1

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss))
        self.logger('Best model path: %s\n' % self.cfg.best_model)

    def configure_optimizers(self):

        self.optimizer = build_optimizer([self.network], self.cfg.optim)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.2, patience=8)
        self.early_stopping = EarlyStopping(**self.cfg.network.early_stopping, trace_func=self.logger)

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='mlp', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)

    def inference_generate(self):

        # torch.set_grad_enabled(False)
        self.network.eval()
        device = self.device

        ds_name = 'test'
        data = self.ds_test

        base_movie_path = os.path.join(self.cfg.results_base_dir, self.cfg.expr_ID)
        num_samples = self.cfg.n_inf_sample

        # mvs = MeshViewers(shape=[1,5])
        previous_movie_name = ''
        for batch_id, batch in enumerate(data):

            movie_name = 's' + self.data_info[ds_name]['frame_names'][batch['idx'].to(torch.long)].split('/s')[-1].replace('/', '_')
            movie_name = movie_name[:np.where([not i.isdigit() for i in movie_name])[0][-1]]
            # if batch_id%100 !=0 :
            if previous_movie_name == movie_name :
                continue
            previous_movie_name = movie_name
            ####

            batch = {k:v.to(self.device) for k,v in batch.items()}

            gender = batch['gender'].data
            if gender == 2:
                sbj_m = self.female_model
            else:
                sbj_m = self.male_model

            sbj_m.v_template = batch['sbj_vtemp'].to(sbj_m.v_template.device)
            # continue

            ### object model

            obj_name = self.data_info[ds_name]['frame_names'][batch['idx'].to(torch.long)].split('/')[-1].split('_')[0]
            # obj_path = os.path.join(self.cfg.datasets.objects_dir, f'{obj_name}.ply')
            obj_path = self.data_info['obj_info'][obj_name]['obj_mesh_file']
            obj_mesh = Mesh(filename=obj_path)
            obj_verts = torch.from_numpy(obj_mesh.v)

            obj_m = ObjectModel(v_template=obj_verts).to(device)

            from tools.gnet_optim import GNetOptim

            fit_smplx = GNetOptim(sbj_model=sbj_m,
                                 obj_model=obj_m,
                                 cfg=self.cfg,
                                 verbose=True)

            sp_anim = sp_animation()


            mov_count = 1 #
            movie_path = os.path.join(base_movie_path, str(mov_count), movie_name+'.html')

            while os.path.exists(movie_path):
                mov_count += 1
                movie_path = os.path.join(base_movie_path, str(mov_count), movie_name+'.html')
            makepath(movie_path, isfile=True)

            grnd_mesh, cage, axis_l = get_ground()
            for i in range(num_samples):
                print(f'{movie_name} -- {i}/{num_samples-1} frames')
                net_output = self.forward(batch)

                optim_output = fit_smplx.fitting(batch, net_output)

                sbj_cnet = Mesh(v=to_cpu(optim_output['cnet_verts'][0]), f=sbj_m.faces, vc=name_to_rgb['pink'])
                sbj_opt = Mesh(v=to_cpu(optim_output['opt_verts'][0]), f=sbj_m.faces, vc=name_to_rgb['green'])
                # obj_i = points_to_spheres(to_cpu(batch['verts_obj'][0]), radius=0.002, vc=name_to_rgb['yellow'])
                obj_i = Mesh(to_cpu(fit_smplx.obj_verts[0]), f = obj_mesh.f, vc=name_to_rgb['yellow'])

                sp_anim.add_frame([sbj_cnet, sbj_opt, obj_i, grnd_mesh], ['course_grasp', 'refined_grasp', 'object', 'ground_mesh'])
            ############################
            sp_anim.save_animation(movie_path)


def train():

    instructions = ''' 
            Please do the following steps before starting the GNet training:
            1. Download GRAB dataset and process GNet dataset using the /data/process_gnet_data.py.
            2. Set the grab_dir and work_dir.
            3. Set the model-path to the folder containing SMPL-X body mdels.
            4. Change the GNet configuration file directly if you want to change the training configs (lr, batch_size, etc).  
                '''
    print(instructions)

    import argparse
    from configs.GNet_config import conf as cfg

    parser = argparse.ArgumentParser(description='GNet-Training')

    parser.add_argument('--work-dir',
                        required=True,
                        type=str,
                        help='The path to the folder to save results')

    parser.add_argument('--grab-path',
                        required=True,
                        type=str,
                        help='The path to the folder that contains GRAB data')

    parser.add_argument('--smplx-path',
                        required=True,
                        type=str,
                        help='The path to the folder containing SMPL-X model downloaded from the website')

    parser.add_argument('--expr-id', default='GNet_V00', type=str,
                        help='Training ID')

    parser.add_argument('--batch-size', default=32, type=int,
                        help='Training batch size')

    parser.add_argument('--num-gpus', default=1,
                        type=int,
                        help='Number of multiple GPUs for training')

    cmd_args = parser.parse_args()

    cfg.expr_ID = cfg.expr_ID if cmd_args.expr_id is None else cmd_args.expr_id

    cfg.datasets.dataset_dir = os.path.join(cmd_args.grab_path,'GNet_data')
    cfg.datasets.grab_path = cmd_args.grab_path
    cfg.body_model.model_path = cmd_args.smplx_path

    cfg.output_folder = cmd_args.work_dir
    cfg.results_base_dir = os.path.join(cfg.output_folder, 'results')
    cfg.num_gpus = cmd_args.num_gpus

    cfg.work_dir = os.path.join(cfg.output_folder, cfg.expr_ID)
    makepath(cfg.work_dir)

    ########################################

    run_trainer_once(cfg)

def run_trainer_once(cfg):

    trainer = Trainer(cfg=cfg)
    OmegaConf.save(trainer.cfg, os.path.join(cfg.work_dir, '{}.yaml'.format(cfg.expr_ID)))

    trainer.fit()

    OmegaConf.save(trainer.cfg, os.path.join(cfg.work_dir, '{}.yaml'.format(cfg.expr_ID)))



if __name__ == '__main__':

    train()
