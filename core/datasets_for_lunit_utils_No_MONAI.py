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

# from monai.data.utils import SUPPORTED_PICKLE_MOD, convert_tables_to_dicts, pickle_hashing
# from monai.transforms import Compose, Randomizable, ThreadUnsafe, Transform, apply_transform, convert_to_contiguous
# from monai.utils import MAX_SEED, deprecated_arg, get_seed, look_up_option, min_version, optional_import
# from monai.utils.misc import first

from torch.utils.data import Dataset
from torchvision import datasets
import os
from PIL import Image
join = os.path.join
# torch.multiprocessing.set_start_method('spawn')

# if TYPE_CHECKING:
#     from tqdm import tqdm

#     has_tqdm = True
# else:
#     tqdm, has_tqdm = optional_import("tqdm", "4.47.0", min_version, "tqdm")

class Dataset_Lunit_for_inference(Dataset):
    def __init__(self, image_paths, transform=None, train=True):   # initial logic happens like transform

        self.img_list = sorted(os.listdir(image_paths))
        # self.device = torch.device('cuda:0')

        if "Thumbs.db" in self.img_list:
            self.img_list.remove("Thumbs.db")
        
        self.image_path = image_paths
        self.transforms = transform

    def __getitem__(self, index):

        image = np.array(Image.open(join(self.image_path, self.img_list[index])))
        image = torch.from_numpy(image).permute((2, 0, 1))

        return image

    def __len__(self):  # return count of sample we have

        return len(self.img_list)



class Dataset_Lunit(Dataset):
    def __init__(self, image_paths, target_paths, transform=None, train=True):   # initial logic happens like transform

        self.img_list = sorted(os.listdir(image_paths))
        self.label_list = sorted(os.listdir(target_paths))
        self.device = torch.device('cuda:0')

        if "Thumbs.db" in self.img_list:
            self.img_list.remove("Thumbs.db")
        if "Thumbs.db" in self.label_list:
            self.label_list.remove("Thumbs.db")
        
        self.image_path = image_paths
        self.label_path = target_paths
        self.transforms = transform

    def __getitem__(self, index):

        image = np.array(Image.open(join(self.image_path, self.img_list[index])))
        label = np.array(Image.open(join(self.label_path, self.label_list[index])))

        image = torch.from_numpy(image).permute((2, 0, 1))
        # image.to(self.device).type(torch.cuda.FloatTensor)
        # image = image / 255 # normalize [0-1]

        label = torch.from_numpy(label).unsqueeze(0)
        # label.to(self.device).type(torch.cuda.FloatTensor)
        # label = label / 255 # normalize [0-1]

        return image, label

    def __len__(self):  # return count of sample we have

        return len(self.img_list)



class Dataset_Lunit_for_3concat(Dataset):
    def __init__(self, image_paths, target_paths, t_path, c_path, transform=None, train=True):   # initial logic happens like transform

        self.img_list = sorted(os.listdir(image_paths))
        self.label_list = sorted(os.listdir(target_paths))
        self.tissue_list = sorted(os.listdir(t_path))
        self.cell_list = sorted(os.listdir(c_path))
        self.device = torch.device('cuda:0')

        if "Thumbs.db" in self.img_list:
            self.img_list.remove("Thumbs.db")
        if "Thumbs.db" in self.label_list:
            self.label_list.remove("Thumbs.db")
        if "Thumbs.db" in self.tissue_list:
            self.tissue_list.remove("Thumbs.db")
        if "Thumbs.db" in self.cell_list:
            self.cell_list.remove("Thumbs.db")
        
        self.image_path = image_paths
        self.label_path = target_paths
        self.t_path = t_path
        self.c_path = c_path
        self.transforms = transform

    def __getitem__(self, index):

        image = np.array(Image.open(join(self.image_path, self.img_list[index])))
        label = np.array(Image.open(join(self.label_path, self.label_list[index])))
        t_img = np.array(Image.open(join(self.t_path, self.tissue_list[index])))
        c_img = np.array(Image.open(join(self.c_path, self.cell_list[index])))

        image = torch.from_numpy(image).permute((2, 0, 1))
        # image.to(self.device).type(torch.cuda.FloatTensor)
        # image = image / 255 # normalize [0-1]

        label = torch.from_numpy(label).unsqueeze(0)
        # label.to(self.device).type(torch.cuda.FloatTensor)
        # label = label / 255 # normalize [0-1]

        t_img = torch.from_numpy(t_img).unsqueeze(0)
        # t_img.to(self.device).type(torch.cuda.FloatTensor)
        # t_img = t_img / 255 # normalize [0-1]

        c_img = torch.from_numpy(c_img).unsqueeze(0)
        # c_img.to(self.device).type(torch.cuda.FloatTensor)
        # c_img = c_img / 255 # normalize [0-1]


        return image, label, t_img, c_img

    def __len__(self):  # return count of sample we have

        return len(self.img_list)




class Dataset_Lunit_for_3concat_for_inference(Dataset):
    def __init__(self, image_paths, target_paths, t_path, transform=None, train=True):   # initial logic happens like transform

        self.img_list = sorted(os.listdir(image_paths))
        self.label_list = sorted(os.listdir(target_paths))
        self.tissue_list = sorted(os.listdir(t_path))
        self.device = torch.device('cuda:0')

        if "Thumbs.db" in self.img_list:
            self.img_list.remove("Thumbs.db")
        if "Thumbs.db" in self.label_list:
            self.label_list.remove("Thumbs.db")
        if "Thumbs.db" in self.tissue_list:
            self.tissue_list.remove("Thumbs.db")

        
        self.image_path = image_paths
        self.label_path = target_paths
        self.t_path = t_path
        self.transforms = transform

    def __getitem__(self, index):

        image = np.array(Image.open(join(self.image_path, self.img_list[index])))
        label = np.array(Image.open(join(self.label_path, self.label_list[index])))
        t_img = np.array(Image.open(join(self.t_path, self.tissue_list[index])))
        # c_img = np.array(Image.open(join(self.c_path, self.cell_list[index])))

        image = torch.from_numpy(image).permute((2, 0, 1))
        # image.to(self.device).type(torch.cuda.FloatTensor)
        # image = image / 255 # normalize [0-1]

        label = torch.from_numpy(label).unsqueeze(0)
        # label.to(self.device).type(torch.cuda.FloatTensor)
        # label = label / 255 # normalize [0-1]

        t_img = torch.from_numpy(t_img).permute((2, 0, 1))
        # t_img.to(self.device).type(torch.cuda.FloatTensor)
        # t_img = t_img / 255 # normalize [0-1]

        # c_img = torch.from_numpy(c_img).permute((2, 0, 1))
        # # c_img.to(self.device).type(torch.cuda.FloatTensor)
        # # c_img = c_img / 255 # normalize [0-1]


        return image, label, t_img # , c_img

    def __len__(self):  # return count of sample we have

        return len(self.img_list)




class Dataset_Lunit_for_3concat_for_inference_with_GT(Dataset):
    def __init__(self, image_paths, target_paths, t_path, b_path, transform=None, train=True):   # initial logic happens like transform

        self.img_list = sorted(os.listdir(image_paths))
        self.label_list = sorted(os.listdir(target_paths))
        self.tissue_list = sorted(os.listdir(t_path))
        self.blobcell_list = sorted(os.listdir(b_path))
        self.device = torch.device('cuda:0')

        if "Thumbs.db" in self.img_list:
            self.img_list.remove("Thumbs.db")
        if "Thumbs.db" in self.label_list:
            self.label_list.remove("Thumbs.db")
        if "Thumbs.db" in self.tissue_list:
            self.tissue_list.remove("Thumbs.db")
        if "Thumbs.db" in self.blobcell_list:
            self.blobcell_list.remove("Thumbs.db")

        
        self.image_path = image_paths
        self.label_path = target_paths
        self.t_path = t_path
        self.b_path = b_path
        self.transforms = transform

    def __getitem__(self, index):

        image = np.array(Image.open(join(self.image_path, self.img_list[index])))
        label = np.array(Image.open(join(self.label_path, self.label_list[index])))
        t_img = np.array(Image.open(join(self.t_path, self.tissue_list[index])))
        b_img = np.array(Image.open(join(self.b_path, self.blobcell_list[index])))
        # c_img = np.array(Image.open(join(self.c_path, self.cell_list[index])))

        image = torch.from_numpy(image).permute((2, 0, 1))

        label = torch.from_numpy(label).unsqueeze(0)
        t_img = torch.from_numpy(t_img).unsqueeze(0)
        b_img = torch.from_numpy(b_img).unsqueeze(0)

        # t_img = torch.from_numpy(t_img).permute((2, 0, 1))
        # b_img = torch.from_numpy(b_img).permute((2, 0, 1))


        return image, label, t_img, b_img

    def __len__(self):  # return count of sample we have

        return len(self.img_list)



# class Dataset_Lunit(Dataset):
#     def __init__(self, image_paths, target_paths, t_path, c_path, transform=None, train=True):   # initial logic happens like transform

#         self.img_list = sorted(os.listdir(image_paths))
#         self.label_list = sorted(os.listdir(target_paths))
#         self.tissue_list = sorted(os.listdir(t_path))
#         self.cell_list = sorted(os.listdir(c_path))

#         if "Thumbs.db" in self.img_list:
#             self.img_list.remove("Thumbs.db")
#         if "Thumbs.db" in self.label_list:
#             self.label_list.remove("Thumbs.db")
#         if "Thumbs.db" in self.tissue_list:
#             self.tissue_list.remove("Thumbs.db")
#         if "Thumbs.db" in self.cell_list:
#             self.cell_list.remove("Thumbs.db")
        
#         self.image_path = image_paths
#         self.label_path = target_paths
#         self.t_path = t_path
#         self.c_path = c_path
#         self.transforms = transform



#     def __getitem__(self, index):

#         image = Image.open(join(self.image_path, self.img_list[index]))
#         mask = Image.open(join(self.label_path, self.label_list[index]))
#         t_img = Image.open(join(self.t_path, self.tissue_list[index]))
#         c_img = Image.open(join(self.c_path, self.cell_list[index]))
#         # t_image = self.transforms(image)
#         t_image = image

#         return t_image, mask, t_img, c_img

#     def __len__(self):  # return count of sample we have

#         return len(self.img_list)









# class Dataset_Lunit(Dataset):
#     def __init__(self, img_dir, annotations_dir, transform=None, target_transform=None):
#         self.img_dir = img_dir
#         self.img_labels = annotations_dir
#         self.transform = transform
#         self.target_transform = target_transform

#     def __len__(self):
#         return len(self.img_labels)

#     def __getitem__(self, idx):
#         img_path = join(self.img_dir, self.img_labels.iloc[idx, 0])
#         image = read_image(img_path)
#         label = self.img_labels.iloc[idx, 1]
#         if self.transform:
#             image = self.transform(image)
#         if self.target_transform:
#             label = self.target_transform(label)
#         return image, label