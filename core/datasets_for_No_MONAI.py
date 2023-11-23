import os
from pathlib import Path
import numpy as np
from PIL import Image
import json
from typing import List

# from util.constants import SAMPLE_SHAPE
from skimage import exposure
import torch
# from torch.utils.data import DataLoader as _TorchDataLoader
# from torch.utils.data import Dataset
from core.datasets_for_lunit_utils_No_MONAI import *
from torch import optim
# from monai.data import *
from torch.utils.data.dataset import ConcatDataset
# from monai.apps import CrossValidation
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

import sys
from abc import ABC, abstractmethod



def call_dataloader_Lunit_for_inference(info, config, data_list):

    ds = Dataset_Lunit_for_inference(
        image_paths=data_list,
        transform=None # ToTensor()
    )

    loader = DataLoader(
        ds, batch_size=1, num_workers=4,
        pin_memory=False, shuffle=False
        # prefetch_factor=10, persistent_workers=False,
    )

    return loader



def call_dataloader_Lunit(info, config, data_list, label_list, mode='train'):
    if mode == 'train' : 
        batch_size = config["BATCH_SIZE"]
        shuffle = True

    elif mode == 'semi' : 
        batch_size = 2
        shuffle = True

    else:
        batch_size = 1
        shuffle = False

    
    ds = Dataset_Lunit(
        image_paths=data_list,
        target_paths = label_list,
        transform=None # ToTensor()
    )

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=info["WORKERS"],
        pin_memory=False, shuffle=shuffle
        # prefetch_factor=10, persistent_workers=False,
    )

    return loader


def call_dataloader_Lunit_3_class(info, config, data_list, label_list, t_data_list, c_data_list, transforms, mode='train'):
    if mode == 'train' : 
        batch_size = config["BATCH_SIZE"]
        shuffle = False

    elif mode == 'semi' : 
        batch_size = 2
        shuffle = True

    else:
        batch_size = 1
        shuffle = False

    
    ds = Dataset_Lunit_for_3concat(
        image_paths=data_list,
        target_paths = label_list,
        t_path=t_data_list,
        c_path=c_data_list,
        transform=None # ToTensor()
    )

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=info["WORKERS"],
        pin_memory=False, shuffle=False
        # prefetch_factor=10, persistent_workers=False,
    )

    return loader




def call_dataloader_Lunit_3_class_for_inference(info, config, data_list, label_list, t_data_list, transforms, mode='train'):
    if mode == 'train' : 
        batch_size = config["BATCH_SIZE"]
        shuffle = False

    elif mode == 'semi' : 
        batch_size = 2
        shuffle = True

    else:
        batch_size = 1
        shuffle = False

    
    ds = Dataset_Lunit_for_3concat_for_inference(
        image_paths=data_list,
        target_paths = label_list,
        t_path=t_data_list,
        transform=None # ToTensor()
    )

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=info["WORKERS"],
        pin_memory=False, shuffle=False
        # prefetch_factor=10, persistent_workers=False,
    )

    return loader



def call_dataloader_Lunit_3_class_for_inference_for_GT(info, config, data_list, label_list, t_data_list, b_data_list, transforms, mode='train'):
    if mode == 'train' : 
        batch_size = config["BATCH_SIZE"]
        shuffle = False

    elif mode == 'semi' : 
        batch_size = 2
        shuffle = True

    else:
        batch_size = 1
        shuffle = False

    
    ds = Dataset_Lunit_for_3concat_for_inference_with_GT(
        image_paths=data_list,
        target_paths = label_list,
        t_path=t_data_list,
        b_path=b_data_list,
        transform=None # ToTensor()
    )

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=info["WORKERS"],
        pin_memory=False, shuffle=False
        # prefetch_factor=10, persistent_workers=False,
    )

    return loader




# class Lunit_DataLoader2(_TorchDataLoader):
#     """This class is meant to load and iterate over the samples
#     already uploaded to GC platform. All cell and tissue samples are
#     concatenated/stacked together sequentially in a single file, one
#     for cell and another for tissue.

#     Parameters
#     ----------
#     cell_path: Path
#         Path to where the cell patches can be found
#     tissue_path: Path
#         Path to where the tissue patches can be found
#     """
#     def __init__(self, cell_path, tissue_path, transform):

#         self.cell_fpath = [os.path.join(cell_path, f) for f in os.listdir(cell_path) if ".png" in f]
#         self.tissue_fpath = [os.path.join(tissue_path, f) for f in os.listdir(tissue_path) if ".png" in f]
#         # assert len(cell_fpath) == len(tissue_fpath) == 1

#         self.cell_patches = np.array(Image.open(self.cell_fpath[0]))
#         self.tissue_patches = np.array(Image.open(self.tissue_fpath[0]))

#         assert (self.cell_patches.shape[1:] == SAMPLE_SHAPE[1:]), \
#             "The same of the input cell patch is incorrect"
#         assert (self.tissue_patches.shape[1:] == SAMPLE_SHAPE[1:]), \
#             "The same of the input tissue patch is incorrect"

#         # Samples are concatenated across the first axis
#         assert self.cell_patches.shape[0] % SAMPLE_SHAPE[0] == 0
#         assert self.tissue_patches.shape[0] % SAMPLE_SHAPE[0] == 0

#         self.num_images = len(self.cell_fpath) # self.cell_patches.shape[0] // SAMPLE_SHAPE[0]

#         # assert self.num_images == self.tissue_patches.shape[0]//SAMPLE_SHAPE[0], \
#         #     "Cell and tissue patches have different number of instances"

#         self.cur_idx = 0

#     def __iter__(self):
#         return self

#     def __next__(self):
#         if self.cur_idx < self.num_images:
#             # Read patch pair and the corresponding id 
#             cell_patch = np.array(Image.open(self.cell_fpath[self.cur_idx]))
#             tissue_patch = np.array(Image.open(self.tissue_fpath[self.cur_idx]))

#             pair_id = self.cur_idx

#             # Increment the current image index
#             self.cur_idx += 1

#             # Return the image data
#             return cell_patch, tissue_patch, pair_id
#         else:
#             # Raise StopIteration when no more images are available
#             raise StopIteration