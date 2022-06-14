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

from typing import NewType, List, Union, Tuple, Optional
from dataclasses import dataclass, fields
import numpy as np
import torch
from yacs.config import CfgNode as CN


__all__ = [
    'CN',
    'Tensor',
    'Array',
    'IntList',
    'IntTuple',
    'IntPair',
    'FloatList',
    'FloatTuple',
    'StringTuple',
    'StringList',
    'TensorTuple',
    'TensorList',
    'DataLoader',
    'BlendShapeDescription',
    'AppearanceDescription',
]


Tensor = NewType('Tensor', torch.Tensor)
Array = NewType('Array', np.ndarray)
IntList = NewType('IntList', List[int])
IntTuple = NewType('IntTuple', Tuple[int])
IntPair = NewType('IntPair', Tuple[int, int])
FloatList = NewType('FloatList', List[float])
FloatTuple = NewType('FloatTuple', Tuple[float])
StringTuple = NewType('StringTuple', Tuple[str])
StringList = NewType('StringList', List[str])

TensorTuple = NewType('TensorTuple', Tuple[Tensor])
TensorList = NewType('TensorList', List[Tensor])

DataLoader = torch.utils.data.DataLoader


@dataclass
class BlendShapeDescription:
    dim: int
    mean: Optional[Tensor] = None

    def keys(self):
        return [f.name for f in fields(self)]

    def __getitem__(self, key):
        for f in fields(self):
            if f.name == key:
                return getattr(self, key)


@dataclass
class AppearanceDescription:
    dim: int
    mean: Optional[Tensor] = None

    def keys(self):
        return [f.name for f in fields(self)]

    def __getitem__(self, key):
        for f in fields(self):
            if f.name == key:
                return getattr(self, key)
