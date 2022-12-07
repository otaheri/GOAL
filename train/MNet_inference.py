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
import mano
import smplx
from smplx import SMPLXLayer


from datetime import datetime

from tools.train_tools import EarlyStopping


from torch import nn, optim

from tensorboardX import SummaryWriter

import glob, time

from psbody.mesh import MeshViewers, Mesh

from psbody.mesh.colors import name_to_rgb
from tools.objectmodel import ObjectModel

from tools.utils import makepath, makelogger, to_cpu, to_np, to_tensor, create_video
from loguru import logger

from tools.train_tools import WeightAnneal
from bps_torch.bps import bps_torch


from omegaconf import OmegaConf

from models.mlp import mnet_model

from losses import build_loss
from optimizers import build_optimizer
from data.mnet_dataloader import LoadData, build_dataloader

from tools.utils import aa2rotmat, rotmat2aa, d62rotmat
from models.model_utils import full2bone, full2bone_aa, parms_6D2full
from tools.train_tools import v2v
from tqdm import tqdm

from tools.utils import LOGGER_DEFAULT_FORMAT

from train.motion_module import motion_module
from tools.vis_tools import sp_animation, get_ground

from tools.mnet_optim import MNetOpt
cdir = os.path.dirname(sys.argv[0])


class Trainer:

    def __init__(self,cfg, inference=False):

        
        self.dtype = torch.float32
        self.cfg = cfg
        self.is_inference = inference

        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)
        logger_path = makepath(os.path.join(cfg.work_dir, 'MNet_test.log'), isfile=True)
        # logger = makelogger(logger_path).info
        # self.logger = logger

        logger.add(logger_path,  backtrace=True, diagnose=True)
        logger.add(lambda x:x,
                   level=cfg.logger_level.upper(),
                   colorize=True,
                   format=LOGGER_DEFAULT_FORMAT
                   )
        self.logger = logger.info
        self.logger('Torch Version: %s\n' % torch.__version__)

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
        self.n_out_frames = self.cfg.network.n_out_frames
        self.network = mnet_model(**cfg.network.mnet_model).to(self.device)

        # Setup the training losses
        self.loss_setup()

        if cfg.num_gpus > 1:
            self.network = nn.DataParallel(self.network)
            self.logger("Training on Multiple GPU's")

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
        vertex_loss_cfg = loss_cfg.get('vertices', {})
        self.vertex_loss_weight = vertex_loss_cfg.get('weight', 0.0)
        self.vertex_loss = build_loss(**vertex_loss_cfg)
        self.logger(f'Vertex loss, weight: {self.vertex_loss},'
                    f' {self.vertex_loss_weight}')

        vertex_consist_loss_cfg = loss_cfg.get('vertices_consist', {})
        self.vertex_consist_loss_weight = vertex_consist_loss_cfg.get('weight', 0.0)
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

        rh_f = torch.from_numpy(np.load(loss_cfg.rh_faces).astype(np.int32)).view(1, -1, 3)
        self.rh_f = rh_f.repeat(self.cfg.datasets.batch_size, 1, 1).to(torch.long)

        self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long)
        self.feet_ids_sampled = to_tensor(np.load(self.cfg.datasets.verts_feet), dtype=torch.long)
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

        ds_name = 'val'
        self.data_info[ds_name] = {}
        ds_val = LoadData(self.cfg.datasets, split_name=ds_name)
        self.data_info[ds_name]['frame_names'] = ds_val.frame_names
        self.data_info[ds_name]['frame_sbjs'] = ds_val.frame_sbjs
        self.data_info[ds_name]['frame_objs'] = ds_val.frame_objs
        self.data_info[ds_name]['chunk_starts'] = np.array(
            [int(fname.split('_')[-1]) for fname in self.data_info[ds_name]['frame_names'][:, 10]]) == 0
        self.ds_val = build_dataloader(ds_val, split=ds_name, cfg=self.cfg.datasets)

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


        self.bps = ds_test.bps
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

        ##############################################

        bs = x['transl'].shape[0]
        pf = self.cfg.network.previous_frames

        dec_x = {}


        dec_x['fullpose'] = x['fullpose_rotmat'][:,11-pf:11,:,:2,:]
        dec_x['transl'] = x['transl'][:,11-pf:11]

        dec_x['betas'] = x['betas']

        verts2last = x['verts'][:, 10:11, self.rh_ids_sampled] - x['verts'][:, -1:, self.rh_ids_sampled]


        if self.use_exp != 0 and self.use_exp != -1:
            dec_x['vel'] = torch.exp(-self.use_exp * x['velocity'][:, 10:11].norm(dim=-1))
            dec_x['verts_to_last_dist'] = torch.exp(-self.use_exp * verts2last.norm(dim=-1))
        else:
            dec_x['vel'] = x['velocity'][:, 10:11].norm(dim=-1)
            dec_x['verts_to_last_dist'] = verts2last.norm(dim=-1)

        dec_x['vel'] = x['velocity'][:, 10:11]
        dec_x['verts'] = x['verts'][:, 10:11]
        dec_x['verts_to_rh'] = verts2last
        dec_x['bps_rh'] = x['bps_rh_glob']

        dec_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in dec_x.values()], dim=1)

        pose, trans, dist, rh2last = self.network(dec_x)

        if self.predict_offsets:
            pose_rotmat = d62rotmat(pose).reshape(bs, self.n_out_frames, -1, 3, 3)
            pose = torch.matmul(pose_rotmat,x['fullpose_rotmat'][:, 10:11])
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

    def train(self):

        self.network.train()
        save_every_it = len(self.ds_train) / self.cfg.summary_steps
        train_loss_dict = {}

        for it, batch in enumerate(self.ds_train):
            batch = {k: batch[k].to(self.device) for k in batch.keys()}

            self.optimizer.zero_grad()

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
                                                    model_name='MNet',
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

                output = self.forward(batch)

                loss_total, losses_dict = self.get_loss(batch, it, output)

                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v.item() for k, v in losses_dict.items()}

            eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}

        return eval_loss_dict

    def get_loss(self, batch, batch_idx, results):


        verts_offset = results['dist']
        rh2last = results['rh2last']
        bparams = results['body_params']

        genders = batch['gender']
        males = genders == 1
        females = ~males

        B, _, _ = batch['transl'].shape
        v_template = batch['sbj_vtemp'].reshape(B, 1, -1, 3).expand(-1, self.n_out_frames, -1, -1)

        FN = sum(females)
        MN = sum(males)

        bparams = {k: v.reshape([-1, self.n_out_frames] + list(v.shape[1:])) for k, v in bparams.items()}

        if FN > 0:
            f_verts_gt = batch['verts'][:, 11:11 + self.n_out_frames][females]

            f_params = {k: v[females].reshape([FN * self.n_out_frames] + list(v.shape[2:])) for k, v in bparams.items()}
            self.female_model.v_template = v_template[females].reshape(-1, 10475, 3)
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices[:, self.verts_ids].reshape(f_verts_gt.shape)

            f_verts_init = batch['verts'][:, 10:11][females]
            f_verts_xyz = verts_offset[females].reshape([FN, -1, 400, 3]) * 0.01 + f_verts_init

        if MN > 0:
            m_verts_gt = batch['verts'][:, 11:11 + self.n_out_frames][males]

            m_params = {k: v[males].reshape([MN * self.n_out_frames] + list(v.shape[2:])) for k, v in bparams.items()}
            self.male_model.v_template = v_template[males].reshape(-1, 10475, 3)
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices[:, self.verts_ids].reshape(m_verts_gt.shape)

            m_verts_init = batch['verts'][:, 10:11][males]
            m_verts_xyz = verts_offset[males].reshape([MN, -1, 400, 3]) * 0.01 + m_verts_init

        losses = {}
        losses_w = {}

        # vertex loss
        if self.vertex_loss_weight > 0:
            losses['vertices'] = 0
            losses['vertices_xyz'] = 0
            if FN > 0:
                losses['vertices'] += self.vertex_loss(f_verts, f_verts_gt)
                losses['vertices_xyz'] += self.vertex_loss(f_verts_xyz, f_verts_gt)
            if MN > 0:
                losses['vertices'] += self.vertex_loss(m_verts, m_verts_gt)
                losses['vertices_xyz'] += self.vertex_loss(m_verts_xyz, m_verts_gt)

            losses_w['vertices'] = losses['vertices']*self.vertex_loss_weight
            losses_w['vertices_xyz'] = losses['vertices_xyz']*(self.vertex_loss_weight+3)


        if self.pose_loss_weight > 0:
            losses['pose'] = self.LossL2(batch['fullpose_rotmat'][:,11:11+self.n_out_frames], bparams['fullpose_rotmat'])
            losses['trans'] = self.LossL1(batch['transl'][:,11:11+self.n_out_frames], bparams['transl'])

            losses_w['pose'] = losses['pose']*self.pose_loss_weight
            losses_w['trans'] = losses['trans']*self.pose_loss_weight/2


        ### right hand vertex loss
        verts2last = batch['verts'][:, 10:11, self.rh_ids_sampled] - batch['verts'][:, -1:, self.rh_ids_sampled]

        if self.rh_vertex_loss_weight > 0:

            losses['rh_vertices'] = 0
            losses['rh_vertices_xyz'] = 0
            if FN > 0:
                rh2rh = verts2last[females].norm(dim=-1).min(dim=-1)[0].reshape(-1)
                w = (rh2rh < .3).to(rh2rh.dtype) + torch.exp(-10*rh2rh)
                w = w.reshape(-1,1,1,1)
                losses['rh_vertices']       += self.vertex_loss(w*f_verts_gt[:, :, self.rh_ids_sampled], w*f_verts[:, :, self.rh_ids_sampled])
                losses['rh_vertices_xyz']   += self.vertex_loss(w*f_verts_gt[:, :, self.rh_ids_sampled], w*f_verts_xyz[:, :, self.rh_ids_sampled])
            if MN > 0:
                rh2rh = verts2last[males].norm(dim=-1).min(dim=-1)[0].reshape(-1)
                w = (rh2rh < .3).to(rh2rh.dtype) + torch.exp(-10 * rh2rh)
                w = w.reshape(-1, 1, 1, 1)
                losses['rh_vertices'] += self.vertex_loss(w * m_verts_gt[:, :, self.rh_ids_sampled], w * m_verts[:, :, self.rh_ids_sampled])
                losses['rh_vertices_xyz'] += self.vertex_loss(w * m_verts_gt[:, :, self.rh_ids_sampled], w * m_verts_xyz[:, :, self.rh_ids_sampled])
            losses_w['rh_vertices'] = losses['rh_vertices']*self.rh_vertex_loss_weight/2
            losses_w['rh_vertices_xyz'] = losses['rh_vertices_xyz']*self.rh_vertex_loss_weight


        if self.contact_loss_weight > 0:
            dist_gt = (batch['verts'][:, 11:11+self.n_out_frames] - batch['verts'][:, -1:])[:,:, self.rh_ids_sampled]
            dist_init = verts2last
            dist_hat = rh2last.reshape(dist_gt.shape)*.01 + dist_init ## get the dist offset in cm

            min_dist = dist_gt.norm(dim=-1).min(dim=-1)[0]
            w = (min_dist < .4).to(min_dist.dtype) + torch.exp(-10 * min_dist)
            w = w.reshape(B, self.n_out_frames, 1, 1)

            losses['dist_sbj2obj'] = self.LossL1(w*dist_hat, w*dist_gt)
            losses_w['dist_sbj2obj'] = self.contact_loss_weight * losses['dist_sbj2obj']

        # feet vertex loss
        if self.feet_vertex_loss_weight > 0:
            feet_verts_gt = batch['verts'][:, 11:11 + self.n_out_frames, self.feet_ids_sampled]

            losses['feet_vertices'] = 0
            losses['feet_vertices_xyz'] = 0
            if FN > 0:
                feet2grnd = feet_verts_gt[females][...,1:2]
                w = 1 + torch.exp(-20 * feet2grnd)
                losses['feet_vertices'] += self.vertex_loss(w * f_verts_gt[:, :, self.feet_ids_sampled], w * f_verts[:, :, self.feet_ids_sampled])
                losses['feet_vertices_xyz'] += self.vertex_loss(w * f_verts_gt[:, :, self.feet_ids_sampled], w * f_verts_xyz[:, :, self.feet_ids_sampled])
            if MN > 0:
                feet2grnd = feet_verts_gt[males][..., 1:2]
                w = 1 + torch.exp(-20 * feet2grnd)
                losses['feet_vertices'] += self.vertex_loss(w * m_verts_gt[:, :, self.feet_ids_sampled], w * m_verts[:, :, self.feet_ids_sampled])
                losses['feet_vertices_xyz'] += self.vertex_loss(w * m_verts_gt[:, :, self.feet_ids_sampled], w * m_verts_xyz[:, :, self.feet_ids_sampled])

            losses_w['feet_vertices'] = losses['feet_vertices']*self.feet_vertex_loss_weight/2
            losses_w['feet_vertices_xyz'] = losses['feet_vertices_xyz']*self.feet_vertex_loss_weight

        with torch.no_grad():
            loss_v2v = []
            loss_v2v_hands = []
            loss_v2v_feet = []

            if FN > 0:
                loss_v2v.append(v2v(f_verts_xyz,f_verts_gt,mean=False))
                loss_v2v_hands.append(v2v(f_verts_xyz[:, :, self.rh_ids_sampled],f_verts_gt[:, :, self.rh_ids_sampled],mean=False))
                loss_v2v_feet.append(v2v(f_verts_xyz[:, :, self.feet_ids_sampled],f_verts_gt[:, :, self.feet_ids_sampled],mean=False))
            if MN > 0:
                loss_v2v.append(v2v(m_verts_xyz, m_verts_gt,mean=False))
                loss_v2v_hands.append(v2v(m_verts_xyz[:, :, self.rh_ids_sampled], m_verts_gt[:, :, self.rh_ids_sampled],mean=False))
                loss_v2v_feet.append(v2v(m_verts_xyz[:, :, self.feet_ids_sampled], m_verts_gt[:, :, self.feet_ids_sampled],mean=False))

            loss_v2v = torch.cat(loss_v2v, dim=0).mean()
            loss_v2v_hands = torch.cat(loss_v2v_hands, dim=0).mean()
            loss_v2v_feet = torch.cat(loss_v2v_feet, dim=0).mean()

        loss_total = torch.stack(list(losses_w.values())).sum()
        losses['loss_total'] = loss_total
        losses['loss_v2v'] = loss_v2v
        losses['loss_v2v_hands'] = loss_v2v_hands
        losses['loss_v2v_feet'] = loss_v2v_feet

        return loss_total, losses

    def set_weight_annealing(self):

        self.vertex_loss_weight_ann = WeightAnneal(start_w=self.vertex_loss_weight/4, end_w=self.vertex_loss_weight, start_batch=0, end_batch=4)
        self.pose_loss_weight_ann   = WeightAnneal(start_w=self.pose_loss_weight, end_w=self.pose_loss_weight/4, start_batch=0, end_batch=4)

        self.contact_loss_weight_ann = WeightAnneal(start_w=self.contact_loss_weight/4, end_w=self.contact_loss_weight, start_batch=0, end_batch=4)

        self.rh_vertex_loss_weight_ann = WeightAnneal(start_w=self.rh_vertex_loss_weight/10, end_w=self.rh_vertex_loss_weight, start_batch=4, end_batch=10)
        self.feet_vertex_loss_weight_ann = WeightAnneal(start_w=self.feet_vertex_loss_weight/10, end_w=self.feet_vertex_loss_weight, start_batch=4, end_batch=10)

    def set_loss_weights(self):
        self.vertex_loss_weight = self.vertex_loss_weight_ann(self.epochs_completed)
        self.pose_loss_weight = self.pose_loss_weight_ann(self.epochs_completed)
        self.contact_loss_weight = self.contact_loss_weight_ann(self.epochs_completed)

        self.rh_vertex_loss_weight = self.rh_vertex_loss_weight_ann(self.epochs_completed)
        self.feet_vertex_loss_weight = self.feet_vertex_loss_weight_ann(self.epochs_completed)

    def fit(self, n_epochs=None, message=None):
        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr = np.inf
        self.set_weight_annealing()

        for epoch_num in range(1, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)

            self.set_loss_weights()

            train_loss_dict = self.train()
            eval_loss_dict  = self.evaluate()


            self.lr_scheduler.step(eval_loss_dict['loss_v2v'])
            cur_lr = self.optimizer.param_groups[0]['lr']

            if cur_lr != prev_lr:
                self.logger('--- learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr

            with torch.no_grad():
                eval_msg = Trainer.create_loss_message(eval_loss_dict, expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                        model_name='MNet',
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

        chunk_starts = self.data_info[ds_name]['chunk_starts']

        visualize = True
        save_meshes = False

        if visualize:
            mvs = MeshViewers()
        else:
            mvs = None

        for batch_id, batch in enumerate(data):

            if not chunk_starts[batch_id]:
                continue

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

            input_data = {k: batch[k].to(self.device) for k in batch.keys()}

            movie_name = 's' + self.data_info[ds_name]['frame_names'][batch['idx'].to(torch.long)][0][:-2].split('/s')[
                -1].replace('/', '_')
            mov_count = 1
            movie_path = os.path.join(base_movie_path, str(mov_count), movie_name + '.html')
            grasp_meshes_path = os.path.join(self.cfg.results_base_dir, f'{obj_name}_grasp')

            print(f' Generated motion for {movie_name}')

            while os.path.exists(movie_path):
                mov_count += 1
                movie_path = os.path.join(base_movie_path, str(mov_count), movie_name + '.html')

            if save_meshes:
                makepath(movie_path, isfile=True)
                makepath(grasp_meshes_path)


            # To transform MNet's generated motion to the current frame's coordinate frame
            moving_b = motion_module(input_data,
                                     sbj_model=sbj_m,
                                     obj_model=obj_m,
                                     cfg=self.cfg)

            moving_b.bps = self.bps
            moving_b.mvs = mvs

            input_data = moving_b.get_current_params()

            # Performs MNet Optimization
            fit_smplx = MNetOpt(sbj_model=sbj_m,
                                 obj_model=obj_m,
                                 cfg=self.cfg,
                                verbose=True)

            fit_smplx.stop = False
            fit_smplx.mvs = mvs

            sp_anim = sp_animation()

            while moving_b.num_iters < 10:
                # print(f'number of iterations: {moving_b.num_iters}')
                net_output = self.forward(input_data)

                fit_results = fit_smplx.fitting(input_data, net_output)
                moving_b(fit_results)
                # moving_b(net_output)

                if fit_smplx.stop:
                    break

                input_data = moving_b.get_current_params()
                min_dist2obj = input_data['verts2obj'][:, 10].reshape(-1, 3).norm(dim=-1).min()
                min_vertex_offset = net_output['dist'].reshape(-1, 3).norm(dim=-1).max()
                min_dist_offset = net_output['rh2last'].reshape(-1, 3).norm(dim=-1).max()

                if min_dist2obj<.003 and min_dist_offset<.2:
                    break


            sbj_params = {k: v.clone() for k, v in moving_b.sbj_params.items()}
            obj_params = {k: v.clone() for k, v in moving_b.obj_params.items()}

            sbj_output_glob = sbj_m(**sbj_params)
            verts_sbj_glob = sbj_output_glob.vertices
            joints_sbj_glob = sbj_output_glob.joints

            obj_out_glob = obj_m(**obj_params)
            verts_obj_glob = obj_out_glob.vertices

            # network_verts = torch.cat(network_verts, dim=0)


            grnd_mesh, cage, axis_l = get_ground()
            #################
            for i in range(moving_b.n_frames - 1):
                sbj_i = Mesh(v=to_cpu(verts_sbj_glob[i]), f=sbj_m.faces, vc=name_to_rgb['pink'])
                obj_i = Mesh(v=to_cpu(verts_obj_glob[i]), f=obj_mesh.f, vc=name_to_rgb['yellow'])

                if visualize:
                    mvs[0][0].set_static_meshes([sbj_i, obj_i])
                    time.sleep(.1)
                if save_meshes:
                    sbj_i.write_ply(grasp_meshes_path + f'/{i:04d}_sbj.ply')
                    obj_i.write_ply(grasp_meshes_path + f'/{i:04d}_obj.ply')

                sp_anim.add_frame([sbj_i, obj_i, grnd_mesh], ['sbj_mesh', 'obj_mesh', 'ground_mesh'])
                ############################
            if save_meshes:
                sp_anim.save_animation(movie_path)
            ############################

def inference():

    instructions = ''' 
                    Please do the following steps before starting the MNet training:
                    1. Download GRAB dataset and process GNet dataset using the /data/process_mnet_data.py.
                    2. Set the grab_path and work_dir.
                    3. Set the model-path to the folder containing SMPL-X body mdels.
                    4. Change the GNet configuration file directly if you want to change the training configs (lr, batch_size, etc).  
                        '''
    print(instructions)

    import argparse

    parser = argparse.ArgumentParser(description='MNet-Inference')


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

    parser.add_argument('--expr-id', default=None, type=str,
                        help='Training ID')


    cmd_args = parser.parse_args()

    if cmd_args.expr_id is None:
        cfg_path = f'{cdir}/../configs/MNet_orig.yaml'
        cfg = OmegaConf.load(cfg_path)
        cfg.best_model = f'{cdir}/../models/MNet_model.pt'
    else:
        expr_ID = cmd_args.expr_id
        work_dir = cmd_args.work_dir
        cfg_path = os.path.join(work_dir,f'{expr_ID}/{expr_ID}.yaml')
        cfg = OmegaConf.load(cfg_path)

    cfg.datasets.dataset_dir = os.path.join(cmd_args.grab_path, 'MNet_data')
    cfg.datasets.grab_path = cmd_args.grab_path
    cfg.body_model.model_path = cmd_args.smplx_path


    cfg.output_folder = cmd_args.work_dir
    cfg.results_base_dir = os.path.join(cfg.output_folder, 'results')
    cfg.work_dir = os.path.join(cfg.output_folder, cfg.expr_ID)

    cfg.batch_size = 1

    tester = Trainer(cfg=cfg, inference=True)

    tester.inference_generate()



if __name__ == '__main__':

    inference()
