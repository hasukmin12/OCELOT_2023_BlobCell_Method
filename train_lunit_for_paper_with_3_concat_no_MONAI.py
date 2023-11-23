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
from core.core_for_lunit_3_concat_No_MONAI import train
# from core.core_for_tanh import train

from core.train_search import main_search
from config.call import call_config
from transforms.call import call_trans_function
# from core.core_for_crossval import train_for_crossval
from core.datasets_for_lunit import *
from core.datasets_for_No_MONAI import *
join = os.path.join

import warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
# torch.multiprocessing.set_start_method('spawn')

# warnings.filterwarnings("ignore")
# torch.multiprocessing.set_start_method('spawn')

def main(info, config, logging=False):
    if logging:
        run = wandb.init(project=info["PROJ_NAME"], entity=info["ENTITY"]) 
        wandb.config.update(config) 

    # torch.manual_seed(config["SEEDS"])
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # np.random.seed(config["SEEDS"])
    
    # Dataset
    # datasets = os.path.join(info["ROOT"], 'dataset.json')

    # train_list = load_decathlon_datalist(datasets, True, 'training')
    # valid_list = load_decathlon_datalist(datasets, True, 'valid')

    # segs = sorted(next(os.walk("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_Tissue_cutting/labelsTr"))[2])
    # val_segs = sorted(next(os.walk("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Lunit_Tissue_cutting/labelsTs"))[2])
    # train_tissue_files = [{"image": join("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/Lunit_Tissue_cutting_Upscaling_255/labelsTr_RGB", seg), 
    #                        "label": join("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/Lunit_Tissue_cutting_Upscaling_255/labelsTr", seg)} for seg in segs]

    # val_tissue_files = [{"image": join("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/Lunit_Tissue_cutting_Upscaling_255/labelsTs_RGB", seg), 
    #                      "label": join("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/Lunit_Tissue_cutting_Upscaling_255/labelsTs", seg)} for seg in val_segs]

    # train_cell_files = [{"image": join("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTr_circle_150_RGB", seg), 
    #                        "label": join("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTr_circle_150", seg)} for seg in segs]

    # val_cell_files = [{"image": join("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTs_circle_150_RGB", seg), 
    #                      "label": join("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTs_circle_150", seg)} for seg in val_segs]


    train_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/images/StainNorm/train/cell"
    train_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/train/cell_14"
    train_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/train/tissue_crop2"
    train_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/train/BlobCell_one_label"

    val_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/images/StainNorm/val/cell"
    val_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/val/cell_14"
    val_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/val/tissue_crop2"
    val_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/pre_augmentation/labels/val/BlobCell_one_label"


    # train_img_path = sorted(os.listdir("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/imagesTr"))
    # train_label_path = sorted(os.listdir("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTr"))
    # train_tissue_path = sorted(os.listdir("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/Lunit_Tissue_cutting_Upscaling_255/labelsTr"))
    # train_cell_path = sorted(os.listdir("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTr_circle_150"))

    # val_img_path = sorted(os.listdir("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/imagesTs"))
    # val_label_path = sorted(os.listdir("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTs"))
    # val_tissue_path = sorted(os.listdir("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/Lunit_Tissue_cutting_Upscaling_255/labelsTs"))
    # val_cell_path = sorted(os.listdir("/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTs_circle_150"))

    # if "Thumbs.db" in train_img_path:
    #     train_img_path.remove("Thumbs.db")
    # if "Thumbs.db" in train_label_path:
    #     train_label_path.remove("Thumbs.db")
    # if "Thumbs.db" in train_tissue_path:
    #     train_tissue_path.remove("Thumbs.db")
    # if "Thumbs.db" in train_cell_path:
    #     train_cell_path.remove("Thumbs.db")

    # if "Thumbs.db" in val_img_path:
    #     val_img_path.remove("Thumbs.db")
    # if "Thumbs.db" in val_label_path:
    #     val_label_path.remove("Thumbs.db")
    # if "Thumbs.db" in val_tissue_path:
    #     val_tissue_path.remove("Thumbs.db")
    # if "Thumbs.db" in val_cell_path:
    #     val_cell_path.remove("Thumbs.db")


    # print('Train', len(train_list), 'Valid', len(valid_list))
    # if logging:
    #     artifact = wandb.Artifact(
    #         "dataset", type="dataset", 
    #         metadata={"train_list":train_list, "valid_list":valid_list, "train_len":len(train_list), "valid_len":len(valid_list)})
    #     run.log_artifact(artifact)

    # train_transforms, val_transforms = call_trans_function(config)

    model = call_model(info, config)
    # model = nn.DataParallel(model)
    if logging:
        wandb.watch(model, log="all")
    model.to("cuda:{0}".format(int(info["GPUS"])))
    # from torchsummary import summary
    # summary(model, (3, 512, 512))
    # print(model)

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
        train_loader = call_dataloader_Lunit_3_class(info, config, train_img_path, train_label_path, train_tissue_path, train_cell_path, transforms=None, mode="train")
        valid_loader = call_dataloader_Lunit_3_class(info, config, val_img_path, val_label_path, val_tissue_path, val_cell_path, transforms=None, mode="valid")
        
        
        # valid_loader = call_dataloader(info, config, valid_list, val_transforms, progress=True, mode="valid")
        # check_data = monai.utils.misc.first(valid_loader)
        # print(
        #     "sanity check:",
        #     check_data["image"].shape,
        #     torch.max(check_data["image"]),
        #     check_data["label"].shape,
        #     torch.max(check_data["label"]),
        # )
    
        while global_step < config["MAX_ITERATIONS"]:
            global_step, dice_val_best, steps_val_best, dice_val_best_class, steps_val_best_class, train_loss = train(
                info, config, global_step, dice_val_best, steps_val_best,
                dice_val_best_class, steps_val_best_class, model, optimizer,
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
    parser.add_argument('-trainer', default= 'Lunit_for_paper_Cell_r14_3_concat_with_one_blobcell') # Stroma_200x_only_Epi
    args = parser.parse_args()
    info, config = call_config(args.trainer)
    warnings.filterwarnings("ignore")

    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, logging=True)

