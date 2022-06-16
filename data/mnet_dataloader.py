
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
import glob
import numpy as np
import torch
from torch.utils import data
from tools.utils import np2torch, torch2np
from tools.utils import to_cpu, to_np, to_tensor

from torch.utils.data.dataloader import default_collate
from omegaconf import DictConfig

from psbody.mesh import Mesh, MeshViewers
from tools.objectmodel import ObjectModel

import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_NUM_WORKERS = {
    'train': 0,
    'val': 0,
    'test': 0
}

class LoadData(data.Dataset):
    def __init__(self,
                 cfg,
                 split_name='train'):

        super().__init__()

        self.split_name = split_name
        self.ds_dir = cfg.dataset_dir
        self.cfg = cfg
        dataset_dir = cfg.dataset_dir

        self.split_name = split_name
        self.ds_dir = dataset_dir

        self.ds = {}
        # dataset_dir = cfg.out_path
        self.ds_path = os.path.join(dataset_dir,split_name)
        datasets = glob.glob(self.ds_path + '/*.npy')

        self.load_ds(datasets)
        # self.normalize()
        self.frame_names = np.load(os.path.join(dataset_dir,split_name, 'frame_names.npz'))['frame_names'].reshape(-1,21)
        self.frame_sbjs = np.asarray([name.split('/')[-2] for name in self.frame_names[:,10]])
        self.frame_st_end = np.asarray([int(name.split('_')[-1]) for name in self.frame_names[:,10]])
        self.frame_objs = np.asarray([os.path.basename(name).split('_')[0] for name in self.frame_names[:,10]])


        self.obj_info = np.load(os.path.join(dataset_dir, 'obj_info.npy'), allow_pickle=True).item()
        self.sbj_info = np.load(os.path.join(dataset_dir, 'sbj_info.npy'), allow_pickle=True).item()

        self.sbjs = np.unique(self.frame_sbjs)

        #######################################

        self.bps = torch.load(os.path.join(dataset_dir, 'bps.pt'))

        ## v_templates
        base_path = os.path.join(self.cfg.grab_path,'tools/subject_meshes/male')

        file_list = []
        for sbj in self.sbjs:
            vt_path = os.path.join(base_path,sbj+'.ply')
            if os.path.exists(vt_path):
                file_list.append(vt_path)
            else:
                file_list.append(vt_path.replace('male','female'))
        self.sbj_vtemp = torch.from_numpy(np.asarray([Mesh(filename=file).v.astype(np.float32) for file in file_list]))
        self.sbj_betas = torch.from_numpy(np.asarray([np.load(file=f.replace('.ply','_betas.npy')).astype(np.float32) for f in file_list]))

        for idx, name in enumerate(self.sbjs):
            self.frame_sbjs[(self.frame_sbjs == name)] = idx

        self.frame_sbjs=torch.from_numpy(self.frame_sbjs.astype(np.int8)).to(torch.long)
        self.ds['frame_sbj_ids'] = self.frame_sbjs

        self.genders = [self.sbj_info[sbj]['gender'] for sbj in self.sbjs]
        self.frame_genders = [self.genders[sbj] for sbj in self.frame_sbjs]
        self.ds['gender'] = torch.Tensor([1 if self.genders[sbj_id] == 'male' else 2 for sbj_id in self.ds['frame_sbj_ids']]).to(torch.long)



        self.objs = list(self.obj_info.keys())
        self.obj_verts = torch.from_numpy(np.asarray([self.obj_info[obj]['verts_sample'].astype(np.float32) for obj in self.objs]))
        for idx, name in enumerate(self.objs):
            self.frame_objs[(self.frame_objs == name)] = idx

        self.frame_objs = torch.from_numpy(self.frame_objs.astype(np.int8)).to(torch.long)


        # find the end of each sequence
        end = self.frame_names.reshape(-1,21)[:,10]
        end_id = np.array([int(n.split('/')[-1].split('_')[-1]) for n in end])
        is_end = loc2vel(end_id,1)
        is_end = is_end < 1.
        is_end[-1] = 1.

        self.is_end = torch.from_numpy(is_end.astype(np.int)).reshape(-1,1)
        self.ds['end'] = self.is_end

    def load_ds(self, dataset_names):
        self.ds = {}
        for name in dataset_names:
            self.ds.update(np.load(name, allow_pickle=True))
        self.ds = np2torch(self.ds)

    def normalize(self):

        norm_data_dir = os.path.join(self.ds_dir,'norm_data.pt')
        if os.path.exists(norm_data_dir):
            self.norm_data = torch.load(norm_data_dir)
        elif self.split_name =='train':
            in_p = {k: (v.mean(0, keepdim=True), v.std(0, keepdim=True) + 1e-10) for k, v in self.ds['in'].items() if v.dtype==torch.float}
            out_p = {k: (v.mean(0, keepdim=True), v.std(0, keepdim=True) + 1e-10) for k, v in self.ds['out'].items()}
            self.norm_data = {'in':in_p, 'out':out_p}
            torch.save(self.norm_data,norm_data_dir)
        else:
            raise('Please run the train split first to normalize the data')

        in_p = self.norm_data['in']
        out_p = self.norm_data['out']

        for k, v in in_p.items():
            self.ds['in'][k] = (self.ds['in'][k]-v[0])/v[1]


    def load_idx(self, idx, source=None):

        if source is None:
            source = self.ds

        out = {}
        for k, v in source.items():
            if isinstance(v, dict):
                out[k] = self.load_idx(idx, v)
            else:
                out[k] = v[idx]
        out['betas'] = self.sbj_betas[self.frame_sbjs[idx]]
        out['sbj_vtemp'] =  self.sbj_vtemp[self.frame_sbjs[idx]]

        velocity = loc2vel(out['verts'], fps=self.cfg.fps)
        out['velocity'] = velocity

        motion_obj = {
            'transl': self.ds['transl_obj'][idx:idx + 1],
            'global_orient': self.ds['global_orient_rotmat_obj'][idx:idx + 1]
        }

        bs = 1
        idx_obj = self.obj_verts[self.frame_objs[idx:idx + 1]]
        obj_m = ObjectModel(v_template=idx_obj,
                            batch_size=bs)
        obj_out = obj_m(**motion_obj, pose2rot=False)
        out['verts_obj'] = obj_out.vertices[0].detach()

        return out

    def __len__(self):
        return self.ds['transl'].shape[0]

    def __getitem__(self, idx):

        data_out = self.load_idx(idx)
        data_out['idx'] = torch.from_numpy(np.array(idx, dtype=np.int32))
        return data_out


def loc2vel(loc,fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B-1))
    # vel = (loc - loc[idxs])/(1/float(fps))
    vel = (loc[1:] - loc[:-1])/(1/float(fps))
    return vel[idxs]


def build_dataloader(dataset: torch.utils.data.Dataset,
                     cfg: DictConfig,
                     split: str = 'train',
                     ) -> torch.utils.data.DataLoader:

    dataset_cfg = cfg
    is_train    = 'train' in split
    is_test    = 'test' in split

    num_workers = dataset_cfg.get('num_workers', DEFAULT_NUM_WORKERS)
    shuffle     = dataset_cfg.get('shuffle', True)

    collate_fn  = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  =   dataset_cfg.batch_size if not is_test else 1,
        num_workers =   num_workers.get(split, 0),
        collate_fn  =   collate_fn,
        drop_last   =   True and (is_train or not is_test),
        pin_memory  =   dataset_cfg.get('pin_memory', False),
        shuffle     =   shuffle and is_train and not is_test,
    )
    return data_loader

