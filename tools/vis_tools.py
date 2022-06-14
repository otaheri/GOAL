
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
import torch
import numpy as np
from psbody.mesh import Mesh, MeshViewers
from psbody.mesh.sphere import Sphere
from psbody.mesh.colors import name_to_rgb
from psbody.mesh.lines import Lines
import scenepic as sp


from tools.train_tools import point2point_signed
from tools.utils import aa2rotmat
from tools.utils import makepath
from tools.utils import to_cpu

def points_to_spheres(points, radius=0.1, vc=name_to_rgb['blue']):

    spheres = Mesh(v=[], f=[])
    for pidx, center in enumerate(points):
        clr = vc[pidx] if len(vc) > 3 else vc
        spheres.concatenate_mesh(Sphere(center, radius).to_mesh(color=clr))
    return spheres

def cage(length=1,vc=name_to_rgb['black']):

    cage_points = np.array([[-1., -1., -1.],
                            [1., 1., 1.],
                            [1., -1., 1.],
                            [-1., 1., -1.]])
    c = Mesh(v=length * cage_points, f=[], vc=vc)
    return c


def create_video(path, fps=30,name='movie'):
    import os
    import subprocess

    src = os.path.join(path,'%*.png')
    movie_path = os.path.join(path,'%s.mp4'%name)
    i = 0
    while os.path.isfile(movie_path):
        movie_path = os.path.join(path,'%s_%02d.mp4'%(name,i))
        i +=1


    cmd = 'ffmpeg -f image2 -r %d -i %s -b:v 6400k -pix_fmt yuv420p %s' % (fps, src, movie_path)
    subprocess.call(cmd.split(' '))
    while not os.path.exists(movie_path):
        continue


def get_ground(cage_size = 7, grnd_size = 5, axis_size = 1):
    ax_v = np.array([[0., 0., 0.],
                     [1.0, 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])
    ax_e = [(0, 1), (0, 2), (0, 3)]

    axis_l = Lines(axis_size*ax_v, ax_e, vc=np.eye(4)[:, 1:])

    g_points = np.array([[-.2, 0.0, -.2],
                         [.2, 0.0, .2],
                         [.2, 0.0, -0.2],
                         [-.2, 0.0, .2]])
    g_faces = np.array([[0, 1, 2], [0, 3, 1]])
    grnd_mesh = Mesh(v=grnd_size * g_points, f=g_faces, vc=name_to_rgb['gray'])

    cage_points = np.array([[-.2, .0, -.2],
                            [.2, .2, .2],
                            [.2, 0., 0.2],
                            [-.2, .2, -.2]])
    cage = [Mesh(v=cage_size * cage_points, f=[], vc=name_to_rgb['black'])]
    return grnd_mesh, cage, axis_l

class sp_animation():
    def __init__(self,
                 width = 1600,
                 height = 1600,
                 ):
        super(sp_animation, self).__init__()

        self.scene = sp.Scene()
        self.main = self.scene.create_canvas_3d(width=width, height=height)
        self.colors = sp.Colors

    def meshes_to_sp(self,meshes_list, layer_names):

        sp_meshes = []



        for i, m in enumerate(meshes_list):
            params = {'vertices' : m.v.astype(np.float32),
                      'normals' : m.estimate_vertex_normals().astype(np.float32),
                      'triangles' : m.f,
                      'colors' : m.vc.astype(np.float32)}
            # params = {'vertices' : m.v.astype(np.float32), 'triangles' : m.f, 'colors' : m.vc.astype(np.float32)}
            # sp_m = sp.Mesh()
            sp_m = self.scene.create_mesh(layer_id = layer_names[i])
            sp_m.add_mesh_with_normals(**params)
            if layer_names[i] == 'ground_mesh':
                sp_m.double_sided=True
            sp_meshes.append(sp_m)

        return sp_meshes

    def add_frame(self,meshes_list_ps, layer_names):

        meshes_list = self.meshes_to_sp(meshes_list_ps, layer_names)
        if not hasattr(self,'focus_point'):
            self.focus_point = meshes_list_ps[1].v.mean(0)
            # center = self.focus_point
            # center[2] = 4
            # rotation = sp.Transforms.rotation_about_z(0)
            # self.camera = sp.Camera(center=center, rotation=rotation, fov_y_degrees=30.0)

        main_frame = self.main.create_frame(focus_point=self.focus_point)
        for i, m in enumerate(meshes_list):
            # self.main.set_layer_settings({layer_names[i]:{}})
            main_frame.add_mesh(m)

    def save_animation(self, sp_anim_name):
        self.scene.link_canvas_events(self.main)
        self.scene.save_as_html(sp_anim_name, title=sp_anim_name.split('/')[-1])