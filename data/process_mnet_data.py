
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

import sys
sys.path.append('')
sys.path.append('..')
import numpy as np
import torch
import os, glob
import smplx
import argparse
import shutil
import time
from datetime import datetime
from tqdm import tqdm

from tools.objectmodel import ObjectModel
from tools.cfg_parser import Config

from tools.utils import makepath, makelogger
from tools.utils import parse_npz
from tools.utils import params2torch
from tools.utils import prepare_params
from tools.utils import to_cpu, to_np, to_tensor
from tools.utils import append2dict
from tools.utils import np2torch, torch2np
from tools.utils import aa2rotmat, rotmat2aa, rotate, rotmul, euler

from bps_torch.bps import bps_torch
from psbody.mesh import Mesh, MeshViewers, MeshViewer


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']

class MNetDataSet(object):

    def __init__(self, cfg, logger=None, **params):

        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.out_path = cfg.out_path
        self.cwd = os.path.dirname(sys.argv[0])
        makepath(self.out_path)

        if logger is None:
            log_dir = os.path.join(self.out_path, 'grab_preprocessing.log')
            self.logger = makelogger(log_dir=log_dir, mode='a').info
        else:
            self.logger = logger
        self.logger('Starting data preprocessing !')

        # assert cfg.intent in INTENTS

        self.intent = cfg.intent
        self.logger('intent:%s --> processing %s sequences!' % (self.intent, self.intent))

        if cfg.splits is None:
            self.splits = { 'test': .1,
                            'val': .05,
                            'train': .85}
        else:
            assert isinstance(cfg.splits, dict)
            self.splits = cfg.splits
            
        self.all_seqs = glob.glob(os.path.join(self.grab_path ,'grab/*/*.npz'))
        
        ### to be filled 
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {'test': [],
                           'val': [],
                           'train': []
                           }

        ### group, mask, and sort sequences based on objects, subjects, and intents
        self.process_sequences()

        self.logger('Total sequences: %d' % len(self.all_seqs))
        self.logger('Selected sequences: %d' % len(self.selected_seqs))
        self.logger('Number of sequences in each data split : train: %d , test: %d , val: %d'
                         %(len(self.split_seqs['train']), len(self.split_seqs['test']), len(self.split_seqs['val'])))

        ### process the data
        self.data_preprocessing(cfg)


    def data_preprocessing(self,cfg):

        self.obj_info = {}
        self.sbj_info = {}

        bps_path = makepath(os.path.join(cfg.out_path, 'bps.pt'), isfile=True)
        bps_orig_path = f'{self.cwd}/../configs/bps.pt'

        # gnet_path = self.out_path.replace('MNet_data', 'GNet_data')
        # gnet_bps_path = os.path.join(gnet_path, 'bps.pt')

        self.bps_torch = bps_torch()



        self.bps = torch.load(bps_orig_path)
        shutil.copy2(bps_orig_path, bps_path)
        self.logger(f'loading bps from {bps_orig_path}')

        # R_bps = torch.tensor(
        #     [[1., 0., 0.],
        #      [0., 0., -1.],
        #      [0., 1., 0.]]).reshape(1, 3, 3).to(device)
        # if os.path.exists(bps_path):
        #     self.bps = torch.load(bps_path)
        #     self.logger(f'loading bps from {bps_path}')
        # elif os.path.exists(gnet_bps_path):
        #     self.bps = torch.load(gnet_bps_path)
        #     shutil.copy2(gnet_bps_path, bps_path)
        #     self.logger(f'loading bps from {gnet_bps_path}')
        # else:
        #     self.bps_obj = sample_sphere_uniform(n_points=cfg.n_obj, radius=cfg.r_obj).reshape(1, -1, 3)
        #     self.bps_sbj = rotate(sample_uniform_cylinder(n_points=cfg.n_sbj, radius=cfg.r_sbj, height=cfg.h_sbj).reshape(1, -1, 3), R_bps.transpose(1, 2))
        #     self.bps_rh = sample_sphere_uniform(n_points=cfg.n_rh, radius=cfg.r_rh).reshape(1, -1, 3)
        #     self.bps_hd = sample_sphere_uniform(n_points=cfg.n_hd, radius=cfg.r_hd).reshape(1, -1, 3)
        #
        #     self.bps = {
        #         'obj':self.bps_obj.cpu(),
        #         'sbj':self.bps_sbj.cpu(),
        #         'rh':self.bps_rh.cpu(),
        #         'hd':self.bps_hd.cpu(),
        #     }
        #     torch.save(self.bps,bps_path)

        vertex_label_contact = to_tensor(np.load(f'{self.cwd}/../consts/vertex_label_contact.npy'), dtype=torch.int8).reshape(1, -1)
        verts_ids = to_tensor(np.load(f'{self.cwd}/../consts/verts_ids_0512.npy'), dtype=torch.long)
        rh_verts_ids = to_tensor(np.load(f'{self.cwd}/../consts/rhand_smplx_ids.npy'), dtype=torch.long)
    
        stime = datetime.now().replace(microsecond=0)
        shutil.copy2(sys.argv[0],
                     os.path.join(self.out_path,
                                  os.path.basename(sys.argv[0]).replace('.py','_%s.py' % datetime.strftime(stime,'%Y%m%d_%H%M'))))

        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}

        for split in self.split_seqs.keys():
            # split = 'train'
            outfname = makepath(os.path.join(cfg.out_path, split, 'grasp_motion_data.npy'), isfile=True)

            if os.path.exists(outfname):
                self.logger('Results for %s split already exist.' % (split))
                continue
            else:
                self.logger('Processing data for %s split.' % (split))

            frame_names = []

            grasp_motion_data = {
                                'transl': [],
                                'fullpose': [],
                                'fullpose_rotmat': [],

                                'verts':[],
                                'velocity':[],

                                'transl_obj': [],
                                'global_orient_obj':[],
                                'global_orient_rotmat_obj': [],

                                'bps_obj_glob':[],
                                'bps_rh_glob':[],

                                'full_seq_id':[],
                                'rel_rot':[],
                                'rel_trans':[],
                                }

            for seq_i, sequence in enumerate(tqdm(self.split_seqs[split])):

                seq_data = parse_npz(sequence)

                obj_name = seq_data.obj_name
                sbj_id   = seq_data.sbj_id

                n_comps  = seq_data.n_comps
                gender   = seq_data.gender

                frame_mask = self.filter_grasp_frames(seq_data)

                T = frame_mask.sum()
                if T < 1:
                    continue # if no frame is selected continue to the next sequence

                ##### motion data preparation
                sbj_vtemp = self.load_sbj_verts(sbj_id, seq_data)
                obj_info = self.load_obj_verts(obj_name, seq_data, cfg.n_verts_sample)

                sbj_params = prepare_params(seq_data.body.params, frame_mask)
                obj_params = prepare_params(seq_data.object.params, frame_mask)
                contact_data_orig = seq_data.contact.body[frame_mask]

                sbj_params_orig = params2torch(sbj_params)
                obj_params_orig = params2torch(obj_params)

                ################# for chunks

                past = self.cfg.past
                future = self.cfg.future

                wind = past + future + 1 + 1

                bs = wind

                with torch.no_grad():
                    sbj_m = smplx.create(model_path=cfg.model_path,
                                         model_type='smplx',
                                         gender=gender,
                                         num_pca_comps=n_comps,
                                         v_template=sbj_vtemp,
                                         batch_size=bs)

                    obj_m = ObjectModel(v_template=obj_info['verts'],
                                        batch_size=bs)

                    root_offset = smplx.lbs.vertices2joints(sbj_m.J_regressor, sbj_m.v_template.view(1, -1, 3))[0, 0]
                    ##### batch motion data selection

                    frames = torch.arange(T).to(torch.long)
                    # duplicate first and last frames to have past and furture frames for them as well
                    frames = torch.cat([torch.zeros(past), frames, torch.ones(future) * (T - 1)]).to(torch.long)
                    chunks = frames.unfold(dimension=0, size=wind-1, step=1)  # create motion chuncks

                    for ch_id, ch in enumerate(chunks):

                        ch = torch.cat([ch, torch.tensor([-1])]).to(torch.long) #to get last frame of motion

                        sbj_params = {k:v[ch] for k,v in sbj_params_orig.items()}
                        obj_params = {k:v[ch] for k,v in obj_params_orig.items()}
                        contact_chunk = contact_data_orig[ch]

                        R = aa2rotmat(sbj_params['global_orient'][past])

                        ############# make relative
                        # transformation from vicon to smplx coordinate frame
                        R_v2s = torch.tensor(
                            [[1., 0., 0.],
                             [0., 0.,-1.],
                             [0., 1., 0.]]).reshape(1,3,3)

                        R_go = rotmul(R_v2s.transpose(1,2),R)

                        ### find transformation to keep z upward
                        z_s = R_go[:, :, 2]
                        z_s[:,1] = 0.
                        z_s = z_s / z_s.norm(dim=-1, keepdim=True)
                        z_w = torch.zeros_like(z_s)
                        z_w[:, 2] = 1.

                        theta = torch.acos(torch.einsum('ij,ij->i', z_w, z_s)).reshape(-1, 1)
                        axis = torch.cross(z_w, z_s)
                        axis = axis / axis.norm(dim=-1, keepdim=True)

                        aa = axis * theta
                        R_aa = aa2rotmat(aa).squeeze()

                        RR = rotmul(R_v2s, R_aa)
                        R_inv = RR.transpose(1,2)

                        motion_sbj,motion_obj, rel_trans =  glob2rel(sbj_params, obj_params, R_inv, root_offset, wind,past)


                        sbj_output = sbj_m(**motion_sbj)
                        verts_sbj = sbj_output.vertices

                        obj_out = obj_m(**motion_obj)
                        verts_obj = obj_out.vertices


                        sbj_in = {k:to_cpu(v.reshape([1,wind]+list(v.shape[1:]))) for k,v in motion_sbj.items()}
                        obj_in = {k+'_obj':to_cpu(v.reshape([1,wind]+list(v.shape[1:]))) for k,v in motion_obj.items()}

                        append2dict(grasp_motion_data,sbj_in)
                        append2dict(grasp_motion_data,obj_in)

                        grasp_motion_data['verts'].append(to_cpu(verts_sbj[:, verts_ids].reshape(1, wind, -1, 3)))
                        grasp_motion_data['rel_rot'].append(to_cpu(R_inv.reshape(1, 3, 3)))
                        grasp_motion_data['rel_trans'].append(to_cpu(rel_trans.reshape(1, 3)))

                        obj_bps = self.bps['obj'] + motion_obj['transl'][-1:]

                        bps_obj = self.bps_torch.encode(x=verts_obj[-1:],
                                                           feature_type=['deltas'],
                                                           custom_basis=obj_bps)['deltas']

                        grasp_motion_data['bps_obj_glob'].append(to_cpu(bps_obj).reshape(1, 1, -1))

                        bps_rh = self.bps_torch.encode(x=verts_sbj[-1:, rh_verts_ids],
                                                       feature_type=['deltas'],
                                                       custom_basis=obj_bps)['deltas']

                        grasp_motion_data['bps_rh_glob'].append(to_cpu(bps_rh).reshape(1, 1, -1))

                        grasp_motion_data['full_seq_id'].append(np.array([seq_i]).reshape(1,-1).astype(int))

                        frame_names.extend(['%s_%s' % (sequence.split('.')[0], fId) for fId in to_cpu(chunks[ch_id])])

            self.logger('Processing for %s split finished' % split)
            self.logger('Total number of frames for %s split is:%d' % (split, len(frame_names)))

            out_data = [grasp_motion_data]
            out_data_name = ['grasp_motion_data']

            import _pickle as pickle
            for idx, _ in enumerate(out_data):
                # data = np2torch(data)
                data_name = out_data_name[idx]
                out_data[idx] = torch2np(out_data[idx])
                outfname = makepath(os.path.join(self.out_path, split, '%s.npy' % data_name), isfile=True)
                pickle.dump(out_data[idx], open(outfname, 'wb'), protocol=4)

            np.savez(os.path.join(self.out_path, split, 'frame_names.npz'), frame_names=frame_names)

            np.save(os.path.join(self.out_path, 'obj_info.npy'), self.obj_info)
            np.save(os.path.join(self.out_path, 'sbj_info.npy'), self.sbj_info)

        # print('hi')
    def process_sequences(self):

        for sequence in self.all_seqs:
            subject_id = sequence.split('/')[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]

            # filter data based on the motion intent

            if 'all' in self.intent:
                pass
            elif 'use' in self.intent and any(intnt in action_name for intnt in INTENTS[:3]):
                continue
            elif all([item not in action_name for item in self.intent]):
                continue

            # group motion sequences based on objects
            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)

            # group motion sequences based on subjects
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)

            # split train, val, and test sequences
            self.selected_seqs.append(sequence)
            if object_name in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif object_name in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if object_name not in self.splits['train']:
                    self.splits['train'].append(object_name)

    def filter_grasp_frames(self,seq_data):

        table_height = seq_data.object.params.transl[0, 2]
        obj_height = seq_data.object.params.transl[:, 2]
        contact_array = seq_data.contact.object
        idxs = np.arange(obj_height.shape[0])
        fil = obj_height > (table_height + .004)

        ## hand velocity
        hand = seq_data.body.params.body_pose[:, 17 * 3:18 * 3]
        hand_rotmat = aa2rotmat(hand)
        hand_ang_vel = loc2vel(hand_rotmat, fps=120).abs().norm(dim=-1).norm(dim=-1).squeeze()

        start_fil = hand_ang_vel > 0.6
        start_frame = idxs[start_fil][2]  # find the first frame of start
        start_fil = idxs > start_frame
        # print(f'skipped {(~start_fil).sum()} frames!\n')

        if fil.sum() < 1:
            return fil
        skip_frame = int(120. / self.cfg.fps)
        fps_fil = (idxs % skip_frame) == 0

        grasp_frame = idxs[fil][0]  # find the first frame of grasp
        grasp_frames = idxs < grasp_frame

        start_pose = seq_data.body.params.fullpose[0:1]

        include_fil = np.isin(contact_array[grasp_frame], cfg.include_joints).any()
        exclude_fil = ~np.isin(contact_array[grasp_frame], cfg.exclude_joints).any()

        grasp_motion_frames = grasp_frames * include_fil * exclude_fil * fps_fil * start_fil

        return grasp_motion_frames

    def load_obj_verts(self, obj_name, seq_data, n_verts_sample=2048):

        mesh_path = os.path.join(self.grab_path, seq_data.object.object_mesh)
        if obj_name not in self.obj_info:
            np.random.seed(100)
            obj_mesh = Mesh(filename=mesh_path)
            verts_obj = np.array(obj_mesh.v)
            faces_obj = np.array(obj_mesh.f)

            if verts_obj.shape[0] > n_verts_sample:
                verts_sample_id = np.random.choice(verts_obj.shape[0], n_verts_sample, replace=False)
            else:
                verts_sample_id = np.arange(verts_obj.shape[0])

            verts_sampled = verts_obj[verts_sample_id]
            self.obj_info[obj_name] = {'verts': verts_obj,
                                       'faces': faces_obj,
                                       'verts_sample_id': verts_sample_id,
                                       'verts_sample': verts_sampled,
                                       'obj_mesh_file': mesh_path}

        return self.obj_info[obj_name]

    def load_sbj_verts(self, sbj_id, seq_data):

        mesh_path = os.path.join(self.grab_path, seq_data.body.vtemp)
        betas_path = mesh_path.replace('.ply', '_betas.npy')
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]['vtemp']
        else:
            sbj_vtemp = np.array(Mesh(filename=mesh_path).v)
            sbj_betas = np.load(betas_path)
            self.sbj_info[sbj_id] = {'vtemp': sbj_vtemp,
                                     'gender': seq_data.gender,
                                     'betas': sbj_betas}
        return sbj_vtemp

def full2bone(pose,trans, expr):

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
                  'transl': trans, 'expression':expr}
    return body_parms

def glob2rel(motion_sbj, motion_obj, R,root_offset, wind, past, rel_trans=None):

    fpose_sbj_rotmat = aa2rotmat(motion_sbj['fullpose'])
    global_orient_sbj_rel = rotmul(R, fpose_sbj_rotmat[:, 0])
    fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

    trans_sbj_rel = rotate((motion_sbj['transl'] + root_offset), R) - root_offset
    trans_obj_rel = rotate(motion_obj['transl'], R)

    global_orient_obj_rotmat = aa2rotmat(motion_obj['global_orient'])
    global_orient_obj_rel = rotmul(global_orient_obj_rotmat, R.transpose(1, 2))

    if rel_trans is None:
        rel_trans = trans_sbj_rel.clone().reshape(wind,-1)[past:past+1]
        rel_trans[:,1] -= rel_trans[:,1]

    motion_sbj['transl'] = to_tensor(trans_sbj_rel) - rel_trans
    motion_sbj['global_orient'] = rotmat2aa(to_tensor(global_orient_sbj_rel).squeeze()).squeeze()
    motion_sbj['global_orient_rotmat'] = to_tensor(global_orient_sbj_rel)
    motion_sbj['fullpose'][:, :3] = motion_sbj['global_orient']
    motion_sbj['fullpose_rotmat'] = fpose_sbj_rotmat

    motion_obj['transl'] = to_tensor(trans_obj_rel) - rel_trans
    motion_obj['global_orient'] = rotmat2aa(to_tensor(global_orient_obj_rel).squeeze()).squeeze()
    motion_obj['global_orient_rotmat'] = to_tensor(global_orient_obj_rel)

    return motion_sbj, motion_obj, rel_trans

def rel2glob(motion_sbj, motion_obj, R, root_offset, T, past, future, rel_trans=None):
    wind = past + future + 1

    fpose_sbj_rotmat = aa2rotmat(motion_sbj['fullpose'])
    global_orient_sbj_rel = rotmul(R, fpose_sbj_rotmat[:, 0])
    fpose_sbj_rotmat[:, 0] = global_orient_sbj_rel

    trans_sbj_rel = rotate((motion_sbj['transl'] + root_offset), R) - root_offset
    trans_obj_rel = rotate(motion_obj['transl'], R)

    global_orient_obj_rotmat = aa2rotmat(motion_obj['global_orient'])
    global_orient_obj_rel = rotmul(global_orient_obj_rotmat, R.transpose(1, 2))

    if rel_trans is None:
        rel_trans = trans_sbj_rel.reshape(T, wind + 1, -1)
        rel_trans = rel_trans[:, past:past + 1].repeat(1, wind + 1, 1).reshape(-1, 3)

    motion_sbj['transl'] = to_tensor(trans_sbj_rel) - rel_trans
    motion_sbj['global_orient'] = rotmat2aa(to_tensor(global_orient_sbj_rel).squeeze()).squeeze()
    motion_sbj['global_orient_rotmat'] = to_tensor(global_orient_sbj_rel)
    motion_sbj['fullpose'][:, :3] = motion_sbj['global_orient']
    motion_sbj['fullpose_rotmat'] = fpose_sbj_rotmat

    motion_obj['transl'] = to_tensor(trans_obj_rel) - rel_trans
    motion_obj['global_orient'] = rotmat2aa(to_tensor(global_orient_obj_rel).squeeze()).squeeze()
    motion_obj['global_orient_rotmat'] = to_tensor(global_orient_obj_rel)

    return motion_sbj, motion_obj, rel_trans

def loc2vel(loc,fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B-1))
    vel = (loc[1:] - loc[:-1])/(1/float(fps))
    return vel[idxs]



if __name__ == '__main__':


    import argparse
    parser = argparse.ArgumentParser(description='MNet-dataset')

    parser.add_argument('--grab-path',
                        required=True,
                        type=str,
                        help='The path to the folder that contains GRAB data')

    parser.add_argument('--smplx-path',
                        required=True,
                        type=str,
                        help='The path to the folder containing SMPL-X model downloaded from the website')

    cmd_args = parser.parse_args()

    grab_path = cmd_args.grab_path
    model_path = cmd_args.smplx_path

    out_path = os.path.join(grab_path, 'MNet_data')

    # split the dataset based on the objects
    grab_splits = {'test': ['mug', 'camera', 'binoculars', 'apple', 'toothpaste'],
                   'val': ['fryingpan', 'toothbrush', 'elephant', 'hand'],
                   'train': []}


    cfg = {

        'intent':['all'], # from 'all', 'use' , 'pass', 'lift' , 'offhand'

        'save_contact': False, # if True, will add the contact info to the saved data
        # motion fps (default is 120.)
        'fps':30.,
        'past':10, #number of past frames to include
        'future':10, #number of future frames to include
        ### splits
        'splits':grab_splits,

        ###IO path
        'grab_path': grab_path,
        'out_path': out_path,

        ### number of vertices samples for each object
        'n_verts_sample': 2048,

        ### body and hand model path
        'model_path':model_path,
        
        ### include/exclude joints
        'include_joints' : list(range(41, 53)),
        # 'required_joints' : [16],  # mouth
        'required_joints' : list(range(53, 56)),  # thumb
        'exclude_joints' : list(range(26, 41)),
        
        ### bps info
        'r_obj' : .15,
        'n_obj': 1024,

        'r_sbj': 1.5,
        'n_sbj': 1024,
        'g_size':20,
        'h_sbj':2.,

        'r_rh': .2,
        'n_rh': 1024,

        'r_hd': .15,
        'n_hd': 2048,

        ### interpolaton params
        'interp_frames':60,
        'fix_length': False,

    }

    cwd = os.getcwd()
    default_cfg_path = os.path.join(cwd, '../configs/grab_preprocessing_cfg.yaml')
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    makepath(cfg.out_path)
    cfg.write_cfg(write_path=cfg.out_path+'/grab_preprocessing_cfg.yaml')

    log_dir = os.path.join(cfg.out_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info

    MNetDataSet(cfg, logger)