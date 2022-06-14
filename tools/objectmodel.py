
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
import torch.nn as nn
from smplx.lbs import batch_rodrigues
from collections import namedtuple

model_output = namedtuple('output', ['vertices', 'global_orient', 'transl'])

class ObjectModel(nn.Module):

    def __init__(self,
                 v_template = None,
                 batch_size = 1,
                 dtype =torch.float32):
        ''' 3D rigid object model

                Parameters
                ----------
                v_template: np.array Vx3, dtype = np.float32
                    The vertices of the object
                batch_size: int, N, optional
                    The batch size used for creating the model variables

                dtype: torch.dtype
                    The data type for the created variables
            '''

        super(ObjectModel, self).__init__()


        self.dtype = dtype

        # Mean template vertices
        if v_template is None:
            v_template = torch.zeros([batch_size, 1000, 3], dtype=dtype)
        elif not torch.is_tensor(v_template):
                v_template = torch.tensor(v_template).reshape(1, -1, 3).to(dtype)
        else:
            v_template = v_template.to(dtype)

        self.register_buffer('v_template', v_template)

        transl = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('transl', nn.Parameter(transl, requires_grad=True))

        global_orient = torch.tensor(np.zeros((batch_size, 3)), dtype=dtype, requires_grad=True)
        self.register_parameter('global_orient', nn.Parameter(global_orient, requires_grad=True))

        self.batch_size = batch_size


    def forward(self, global_orient=None, transl=None, v_template=None, pose2rot= True, **kwargs):

        ''' Forward pass for the object model

        Parameters
            ----------
            global_orient: torch.tensor, optional, shape Bx3
                If given, ignore the member variable and use it as the global
                rotation of the body. Useful if someone wishes to predicts this
                with an external model. (default=None)

            transl: torch.tensor, optional, shape Bx3
                If given, ignore the member variable `transl` and use it
                instead. For example, it can used if the translation
                `transl` is predicted from some external model.
                (default=None)
            v_template: torch.tensor, optional, shape BxVx3
                The new object vertices to overwrite the default vertices

        Returns
            -------
                output: ModelOutput
                A named tuple of type `ModelOutput`
        '''

        device = self.v_template.device

        model_vars = [global_orient, transl]
        batch_size = 1
        for var in model_vars:
            if var is None:
                continue
            batch_size = max(batch_size, len(var))


        if global_orient is None:
            pose2rot = False
            global_orient = batch_rodrigues(self.global_orient.view(-1, 3)).reshape([batch_size, 3, 3])
        if transl is None:
            transl = self.transl.view(-1, 1, 3).expand(batch_size, 1, 3).contiguous()

        if v_template is None:
            v_template = self.v_template

        if pose2rot:
            global_orient = batch_rodrigues(global_orient.view(-1, 3)).reshape([batch_size, 3, 3])

        transl = transl.reshape(-1, 1, 3)
        global_orient = global_orient.reshape(-1, 3, 3)

        if v_template.ndim < 3:
            v_template = v_template.reshape(1, -1, 3)


        vertices = torch.matmul(v_template, global_orient) + transl

        output = model_output(vertices=vertices,
                              global_orient=global_orient,
                              transl=transl)

        return output

