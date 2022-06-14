
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


import numpy as np
import torch
import logging
from copy import copy

import torch.nn.functional as F
import pytorch3d.transforms as t3d
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import point_face_distance, face_point_distance


LOGGER_DEFAULT_FORMAT = ('<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |'
                  ' <level>{level: <8}</level> |'
                  ' <cyan>{name}</cyan>:<cyan>{function}</cyan>:'
                  '<cyan>{line}</cyan> - <level>{message}</level>')



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_cpu = lambda tensor: tensor.detach().cpu().numpy()

def parse_npz(npz, allow_pickle=True):
    npz = np.load(npz, allow_pickle=allow_pickle)
    npz = {k: npz[k].item() for k in npz.files}
    return DotDict(npz)

def params2torch(params, dtype = torch.float32):
    return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}

def prepare_params(params, frame_mask, rel_trans = None, dtype = np.float32):
    n_params = {k: v[frame_mask].astype(dtype)  for k, v in params.items()}
    if rel_trans is not None:
        n_params['transl'] -= rel_trans
    return n_params


def append2dict(source, data):
    for k in data.keys():
        if k in source.keys():
            if isinstance(data[k], list):
                source[k] += data[k]
            else:
                source[k].append(data[k])

def np2torch(item):
    out = {}
    for k, v in item.items():
        if v == []:
            continue
        if isinstance(v, list):
            try:
                out[k] = torch.from_numpy(np.concatenate(v))
            except:
                out[k] = torch.from_numpy(np.array(v))
        elif isinstance(v, dict):
            if v=={}:
                continue
            out[k] = np2torch(v)
        else:
            out[k] = torch.from_numpy(v)
    return out

def torch2np(item):
    out = {}
    for k, v in item.items():
        if v ==[] or v=={}:
            continue
        if isinstance(v, list):
            try:
                out[k] = np.array(np.concatenate(v))
            except:
                out[k] = np.array(np.array(v))
        elif isinstance(v, dict):
            out[k] = torch2np(v)
        else:
            out[k] = np.array(v)
    return out

def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array

def batch_to(dst, *args):
    return [x.to(dst) if x is not None else None for x in args]


def loc2vel(loc,fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B-1))
    vel = (loc - loc[idxs])/(1/float(fps))
    return vel

# def loc2vel(loc,fps):
#     B = loc.shape[0]
#     idxs = [0] + list(range(B-1))
#     # vel = (loc - loc[idxs])/(1/float(fps))
#     vel = (loc[1:] - loc[:-1])/(1/float(fps))
#     return vel[idxs]

def vel2acc(vel,fps):
    B = vel.shape[0]
    idxs = [0] + list(range(B - 1))
    acc = (vel - vel[idxs]) / (1 / float(fps))
    return acc

def loc2acc(loc,fps):
    vel = loc2vel(loc,fps)
    acc = vel2acc(vel,fps)
    return acc, vel


def d62rotmat(pose):
    pose = to_tensor(pose)
    reshaped_input = pose.reshape(-1, 6)
    return t3d.rotation_6d_to_matrix(reshaped_input)

def rotmat2d6(pose):
    pose = to_tensor(pose)
    reshaped_input = pose.reshape(-1, 3, 3)
    return t3d.matrix_to_rotation_6d(reshaped_input)

def aa2rotmat(pose):
    pose = to_tensor(pose)
    shape = pose.shape
    if len(shape) < 2:
        pose = pose.unsqueeze(dim=0)
    T = pose.shape[0]
    reshaped_input = pose.reshape(-1, 3)
    return t3d.axis_angle_to_matrix(reshaped_input).view(T, -1, 3, 3)

def rotmat2aa(pose):
    pose = to_tensor(pose)
    reshaped_input = pose.reshape(-1, 3, 3)
    quat = t3d.matrix_to_quaternion(reshaped_input)
    return t3d.quaternion_to_axis_angle(quat)

def euler(rots, order='xyz', units='deg'):

    rots = np.asarray(rots)
    single_val = False if len(rots.shape)>1 else True
    rots = rots.reshape(-1,3)
    rotmats = []

    for xyz in rots:
        if units == 'deg':
            xyz = np.radians(xyz)
        r = np.eye(3)
        for theta, axis in zip(xyz,order):
            c = np.cos(theta)
            s = np.sin(theta)
            if axis=='x':
                r = np.dot(np.array([[1,0,0],[0,c,-s],[0,s,c]]), r)
            if axis=='y':
                r = np.dot(np.array([[c,0,s],[0,1,0],[-s,0,c]]), r)
            if axis=='z':
                r = np.dot(np.array([[c,-s,0],[s,c,0],[0,0,1]]), r)
        rotmats.append(r)
    rotmats = np.stack(rotmats).astype(np.float32)
    if single_val:
        return rotmats[0]
    else:
        return rotmats

def batch_euler(bxyz,order='xyz', units='deg'):

    br = []
    for frame in range(bxyz.shape[0]):
        br.append(euler(bxyz[frame], order, units))
    return np.stack(br).astype(np.float32)

def rotate(points,R):
    shape = list(points.shape)
    points = to_tensor(points)
    R = to_tensor(R)
    if len(shape)>3:
        points = points.squeeze()
    if len(shape)<3:
        points = points.unsqueeze(dim=1)
    if R.shape[0] > shape[0]:
        shape[0] = R.shape[0]
    r_points = torch.matmul(points, R.transpose(1,2))
    return r_points.reshape(shape)

def rotmul(rotmat,R):

    if rotmat.ndim>3:
        rotmat = to_tensor(rotmat).squeeze()
    if R.ndim>3:
        R = to_tensor(R).squeeze()
    rot = torch.matmul(rotmat, R)
    return rot


smplx_parents =[-1,  0,  0,  0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  9,  9, 12, 13, 14,
                16, 17, 18, 19, 15, 15, 15, 20, 25, 26, 20, 28, 29, 20, 31, 32, 20, 34,
                35, 20, 37, 38, 21, 40, 41, 21, 43, 44, 21, 46, 47, 21, 49, 50, 21, 52,
                53]
def smplx_loc2glob(local_pose):

    bs = local_pose.shape[0]
    local_pose = local_pose.view(bs, -1, 3, 3)
    global_pose = local_pose.clone()

    for i in range(1,len(smplx_parents)):
        global_pose[:,i] = torch.matmul(global_pose[:, smplx_parents[i]], global_pose[:, i].clone())

    return global_pose.reshape(bs,-1,3,3)



def makepath(desired_path, isfile = False):
    '''
    if the path does not exist make it
    :param desired_path: can be path to a file or a folder name
    :return:
    '''
    import os
    if isfile:
        if not os.path.exists(os.path.dirname(desired_path)):os.makedirs(os.path.dirname(desired_path))
    else:
        if not os.path.exists(desired_path): os.makedirs(desired_path)
    return desired_path

def makelogger(log_dir,mode='a'):

    makepath(log_dir, isfile=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch.setFormatter(formatter)

    logger.addHandler(ch)

    fh = logging.FileHandler('%s'%log_dir, mode=mode)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    return logger


def DotDict(in_dict):

    out_dict = copy(in_dict)
    for k,v in out_dict.items():
       if isinstance(v,dict):
           out_dict[k] = DotDict(v)
    return dotdict(out_dict)

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def create_video(path, fps=30,name='movie'):
    import os
    import subprocess

    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'../%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'../%s_%02d.mp4'%(name,i))
        i +=1

    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)

    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue



def point2surface(meshes: Meshes, pcls: Pointclouds):
    """
    Computes the distance between a pointcloud and a mesh within a batch.
    Given a pair `(mesh, pcl)` in the batch, we define the distance to be the
    sum of two distances, namely `point_face(mesh, pcl) + face_point(mesh, pcl)`

    `point_face(mesh, pcl)`: Computes the squared distance of each point p in pcl
        to the closest triangular face in mesh and averages across all points in pcl
    `face_point(mesh, pcl)`: Computes the squared distance of each triangular face in
        mesh to the closest point in pcl and averages across all faces in mesh.

    The above distance functions are applied for all `(mesh, pcl)` pairs in the batch
    and then averaged across the batch.

    Args:
        meshes: A Meshes data structure containing N meshes
        pcls: A Pointclouds data structure containing N pointclouds

    Returns:
        loss: The `point_face(mesh, pcl) + face_point(mesh, pcl)` distance
            between all `(mesh, pcl)` in a batch averaged across the batch.
    """

    if len(meshes) != len(pcls):
        raise ValueError("meshes and pointclouds must be equal sized batches")
    N = len(meshes)

    # packed representation for pointclouds
    points = pcls.points_packed()  # (P, 3)
    points_first_idx = pcls.cloud_to_packed_first_idx()
    max_points = pcls.num_points_per_cloud().max().item()

    # packed representation for faces
    verts_packed = meshes.verts_packed()
    faces_packed = meshes.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = meshes.mesh_to_faces_packed_first_idx()
    max_tris = meshes.num_faces_per_mesh().max().item()

    # faces = meshes.faces_packed()
    # v0, v1, v2 = faces.chunk(3, dim=1)
    # e01 = torch.cat([verts_packed[v0], verts_packed[v1]], dim=1)  # (sum(F_n), 2)
    # e12 = torch.cat([verts_packed[v1], verts_packed[v2]], dim=1)  # (sum(F_n), 2)
    # e20 = torch.cat([verts_packed[v2], verts_packed[v0]], dim=1)  # (sum(F_n), 2)
    #
    # e0 = (e01[:,0] - e01[:,1]).norm(dim=1)>.001
    # e1 = (e12[:,0] - e12[:,1]).norm(dim=1)>.001
    # e2 = (e20[:,0] - e20[:,1]).norm(dim=1)>.001
    #
    # tris = verts_packed[faces_packed[e0*e1*e2]]

    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    return point_to_face.reshape(N,-1)
