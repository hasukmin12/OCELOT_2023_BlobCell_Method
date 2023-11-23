import os, glob
import torch
from torch import optim
from monai.data import *
from torch.utils.data.dataset import ConcatDataset
from monai.apps import CrossValidation

import sys
from abc import ABC, abstractmethod
class CVDataset(ABC, CacheDataset):
    """
    Base class to generate cross validation datasets.

    """
    def __init__(
        self,
        data,
        transform,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=4,
    ) -> None:
        data = self._split_datalist(datalist=data)
        CacheDataset.__init__(
            self, data, transform, cache_num=cache_num, cache_rate=cache_rate, num_workers=num_workers
        )

    @abstractmethod
    def _split_datalist(self, datalist):
        raise NotImplementedError(f"Subclass {self.__class__.__name__} must implement this method.")



def call_fold_dataset(list_, target_fold, total_folds=5):
    train, valid = [],[]
    count = 0
    for i in list_:
        count += 1
        if count == total_folds: count = 1
        if count == target_fold:
            valid.append(i)
        else:
            train.append(i)
    return train, valid

def call_dataloader(info, config, data_list, transforms, progress=False, mode='train'):
    if mode == 'train' : 
        batch_size = config["BATCH_SIZE"]
        shuffle = False

    elif mode == 'semi' : 
        batch_size = 2
        shuffle = True

    else:
        batch_size = 1
        shuffle = False

    if info["MEM_CACHE"]>0:
        ds = CacheDataset(
            data=data_list,
            transform=transforms,
            cache_rate=info["MEM_CACHE"], num_workers=info["WORKERS"],
            progress=progress
        )
    else:
        ds = Dataset(
            data=data_list,
            transform=transforms,
        )

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=info["WORKERS"],
        pin_memory=False, shuffle=shuffle 
        # prefetch_factor=10, persistent_workers=False,
    )

    return loader













def call_dataloader_lunit(info, config, data_list, tissue_data_list, transforms, progress=False, mode='train'):
    if mode == 'train' : 
        batch_size = config["BATCH_SIZE"]
        shuffle = False

    elif mode == 'semi' : 
        batch_size = 2
        shuffle = True

    else:
        batch_size = 1
        shuffle = False

    if info["MEM_CACHE"]>0:
        ds = CacheDataset(
            data=data_list,
            transform=transforms,
            cache_rate=info["MEM_CACHE"], num_workers=info["WORKERS"],
            progress=progress
        )
    else:
        ds = Dataset(
            data=data_list,
            t_data=tissue_data_list,
            transform=transforms,
        )

    loader = DataLoader(
        ds, batch_size=batch_size, num_workers=info["WORKERS"],
        pin_memory=False, shuffle=shuffle 
        # prefetch_factor=10, persistent_workers=False,
    )

    return loader





















def call_dataloader_Crossvalidation(info, config, data_list, transforms, progress=False, mode='train'):
    num = info["FOLD_for_CrossValidation"]
    folds = list(range(num))

    cvdataset = CrossValidation(
        dataset_cls=CVDataset,
        data=data_list,
        nfolds=5,
        # seed=12345,
        transform=transforms,
    )
    
    if mode == 'train' : 
        batch_size = config["BATCH_SIZE"]
        shuffle = True
        ds = [cvdataset.get_dataset(folds=folds[0: i] + folds[(i + 1):]) for i in folds]
        loader = [DataLoader(ds[i], batch_size=batch_size, shuffle=shuffle, num_workers=info["WORKERS"]) for i in folds]

    else: # val
        batch_size = 1
        shuffle = False
        ds = [cvdataset.get_dataset(folds=i, transform=transforms) for i in range(num)]
        loader = [DataLoader(ds[i], batch_size=1, shuffle=shuffle, num_workers=info["WORKERS"]) for i in folds]

    return loader



# def call_dataloader_for_semi(info, config, data_list, unlabel_list, transforms, transforms_for_unlabel, progress=False, mode='train'):
#     if mode == 'train' : 
#         batch_size = config["BATCH_SIZE"]
#         shuffle = True

#     elif mode == 'semi' : 
#         batch_size = 2
#         shuffle = True

#     else:
#         batch_size = 1
#         shuffle = False

#     if info["MEM_CACHE"]>0:
#         train_ds = CacheDataset(
#             data=data_list,
#             transform=transforms,
#             cache_rate=info["MEM_CACHE"], num_workers=info["WORKERS"],
#             progress=progress
#         )
#         un_ds = CacheDataset(
#             data=unlabel_list,
#             transform=transforms_for_unlabel,
#             cache_rate=info["MEM_CACHE"], num_workers=info["WORKERS"],
#             progress=progress
#         )
#     else:
#         train_ds = Dataset(
#             data=data_list,
#             transform=transforms,
#         )
#         un_ds = Dataset(
#             data=unlabel_list,
#             transform=transforms_for_unlabel,
#         )


#     concat_ds = ConcatDataset([train_ds, un_ds])

#     loader = DataLoader(
#         concat_ds, batch_size=batch_size, num_workers=info["WORKERS"],
#         pin_memory=True, shuffle=shuffle 
#         # prefetch_factor=10, persistent_workers=False,
#     )

#     return loader
    

