# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import collections.abc
import math
import pickle
import shutil
import sys
import tempfile
import threading
import time
import warnings
from copy import copy, deepcopy
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import IO, TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch.serialization import DEFAULT_PROTOCOL
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset

from monai.data.utils import SUPPORTED_PICKLE_MOD, convert_tables_to_dicts, pickle_hashing
from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous
from monai.utils import MAX_SEED, deprecated_arg, get_seed, look_up_option, min_version, optional_import
from monai.utils.misc import first

if TYPE_CHECKING:
    from tqdm import tqdm

    has_tqdm = True
else:
    tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

lmdb, _ = optional_import("lmdb")
pd, _ = optional_import("pandas")


class Dataset_Lunit(_TorchDataset):
    """
    A generic dataset with a length property and an optional callable data transform
    when fetching a data sample.
    If passing slicing indices, will return a PyTorch Subset, for example: `data: Subset = dataset[1:4]`,
    for more details, please check: https://pytorch.org/docs/stable/data.html#torch.utils.data.Subset

    For example, typical input data can be a list of dictionaries::

        [{                            {                            {
             'img': 'image1.nii.gz',      'img': 'image2.nii.gz',      'img': 'image3.nii.gz',
             'seg': 'label1.nii.gz',      'seg': 'label2.nii.gz',      'seg': 'label3.nii.gz',
             'extra': 123                 'extra': 456                 'extra': 789
         },                           },                           }]
    """

    def __init__(self, data: Sequence, t_data: Sequence, c_data: Sequence, transform: Optional[Callable] = None, v_transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.

        """
        self.data = data
        self.t_data = t_data
        self.c_data = c_data
        self.transform = transform
        self.v_transform = transform
        # self.c_transform = c_transform

    def __len__(self) -> int:
        return len(self.data)

    def _transform(self, index: int):
        """
        Fetch single data item from `self.data`.
        """
        data_i = self.data[index]
        t_data = self.t_data[index]
        c_data = self.c_data[index]
        return apply_transform(self.transform, data_i) if self.transform is not None else data_i, \
            apply_transform(self.transform, t_data) if self.transform is not None else t_data,\
            apply_transform(self.transform, c_data) if self.transform is not None else c_data
    
    # def _vtransform(self, index: int):
    #     """
    #     Fetch single data item from `self.data`.
    #     """
    #     t_data = self.t_data[index]
    #     return apply_transform(self.v_transform, t_data) if self.v_transform is not None else t_data
    

    # def _ctransform(self, index: int):
    #     """
    #     Fetch single data item from `self.data`.
    #     """
    #     c_data = self.c_data[index]
    #     return apply_transform(self.transform, c_data) if self.transform is not None else c_data


    def __getitem__(self, index: Union[int, slice, Sequence[int]]):
        """
        Returns a `Subset` if `index` is a slice or Sequence, a data item otherwise.
        """
        if isinstance(index, slice):
            # dataset[:42]
            start, stop, step = index.indices(len(self))
            indices = range(start, stop, step)
            return Subset(dataset=self, indices=indices)
        if isinstance(index, collections.abc.Sequence):
            # dataset[[1, 3, 4]]
            return Subset(dataset=self, indices=index)
        return self._transform(index) # , self._vtransform(index) # , self._ctransform(index)

