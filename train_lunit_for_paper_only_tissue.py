import os, tempfile
import random
import torch
import wandb
import numpy as np
import argparse as ap
import torch.nn as nn
import monai
from core.call_data import *
from core.call_model import *
from core.core_for_lunit_paper_No_MONAI import train
# from core.core_for_tanh import train

from core.train_search import main_search
from config.call import call_config
from transforms.call import call_trans_function
# from core.core_for_crossval import train_for_crossval
from core.datasets_for_No_MONAI import call_dataloader_Lunit

import warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# from medpalm.model import MedPalm

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# CUDA_VISIBLE_DEVICES = '1'

# warnings.filterwarnings("ignore")
# torch.multiprocessing.set_start_method('spawn')

def main(info, config, logging=False):
    if logging:
        run = wandb.init(project=info["PROJ_NAME"], entity=info["ENTITY"]) 
        wandb.config.update(config) 

    torch.manual_seed(config["SEEDS"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["SEEDS"])
    
    # Dataset
    train_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/images/StainNorm/train/tissue"
    train_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/train/tissue2"
   
    val_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/images/StainNorm/val/tissue"
    val_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/val/tissue2"


    train_list = sorted(os.listdir(train_img_path))
    val_list = sorted(os.listdir(val_img_path))
    print('Train', len(train_list), 'Valid', len(val_list))

    model = call_model(info, config)
    # model = nn.DataParallel(model)
    if logging:
        wandb.watch(model, log="all")
    # model.to("cuda:1")
    # print("cuda:{0}".format(int(info["GPUS"])))
    model.to("cuda:{0}".format(int(info["GPUS"])))

    optimizer = call_optimizer(config, model)

    if config["LOAD_MODEL"]:
        check_point = torch.load(os.path.join(info["LOGDIR"], f"model_best.pth"))
        try:
            model.load_state_dict(check_point['model_state_dict'])
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
        except:
            model.load_state_dict(check_point)

    best_loss = 1.
    global_step = 0
    dice_val_best = 0.0
    steps_val_best = 0.0

    dice_val_best_class = 0.0
    steps_val_best_class = 0.0
    



    if info["FOLD_for_CrossValidation"] == False:
        train_loader = call_dataloader_Lunit(info, config, train_img_path, train_label_path, mode="train")
        valid_loader = call_dataloader_Lunit(info, config, val_img_path, val_label_path, mode="valid")
        # check_data = monai.utils.misc.first(train_loader)
        # print(
        #     "sanity check:",
        #     check_data["image"].shape,
        #     torch.max(check_data["image"]),
        #     check_data["label"].shape,
        #     torch.max(check_data["label"]),
        # )
    
        while global_step < config["MAX_ITERATIONS"]:
            global_step, dice_val_best, steps_val_best, dice_val_best_class, steps_val_best_class, train_loss = train(
                info, config, global_step, dice_val_best, steps_val_best, dice_val_best_class, steps_val_best_class,
                model, optimizer,
                train_loader, valid_loader, 
                logging, deep_supervision=info["Deep_Supervision"],
            )   
            if logging:
                wandb.log({
                    'train_loss': train_loss,
                    # 'train_dice': train_dice,
                }) 
    

    # else:
    #     train_loader = call_dataloader_Crossvalidation(info, config, train_list, train_transforms, progress=True, mode="train")
    #     valid_loader = call_dataloader_Crossvalidation(info, config, valid_list, val_transforms, progress=True, mode="valid")

    #     check_data = monai.utils.misc.first(train_loader[0])
    #     print(
    #         "sanity check:",
    #         check_data["image"].shape,
    #         torch.max(check_data["image"]),
    #         check_data["label"].shape,
    #         torch.max(check_data["label"]),
    #     )

    #     while global_step < config["MAX_ITERATIONS"]:
    #         global_step, dice_val_best, steps_val_best, train_loss = train_for_crossval(
    #             info, config, global_step, dice_val_best, steps_val_best,
    #             model, optimizer,
    #             train_loader, valid_loader, 
    #             logging, deep_supervision=info["Deep_Supervision"],
    #         )   
    #         if logging:
    #             wandb.log({
    #                 'train_loss': train_loss,
    #                 # 'train_dice': train_dice,
    #             })



    if logging:
        artifact = wandb.Artifact('model', type='model')
        # artifact.add_file(
        #     os.path.join(info["LOGDIR"], f"model_best.pth"), 
        #     name=f'model/{config["MODEL_NAME"]}')
        run.log_artifact(artifact)
    return 

    

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('-trainer', default= 'Lunit_for_paper_Tissue_Only_StainNorm') # Stroma_200x_only_Epi
    args = parser.parse_args()
    info, config = call_config(args.trainer)
    warnings.filterwarnings("ignore")

    # os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    # CUDA_VISIBLE_DEVICES = "1"
    main(info, config, logging=True)

