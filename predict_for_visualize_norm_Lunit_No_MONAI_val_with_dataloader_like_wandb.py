#%%
import os
from requests import post
import torch
import wandb
import argparse as ap
import numpy as np
# import nibabel as nib
from tqdm import tqdm
from typing import Tuple
from PIL import Image

from monai.inferers import sliding_window_inference
from monai.data import *
from monai.transforms import *
#from monai.handlers.utils import from_engine

from core.utils import *
from core.call_model import *
from config.call import call_config
# from skimage import io, segmentation, morphology, measure, exposure
import time
import tifffile as tif
# import cv2
import cv2
import PIL
import matplotlib.pyplot as plt

from evaluation.evaluator import evaluate_folder
from monai.data import Dataset, DataLoader, create_test_image_3d, decollate_batch

from core.datasets_for_lunit import call_dataloader_Lunit
from core.datasets_for_lunit_utils import *
from copy import deepcopy
from skimage import io, segmentation, morphology, measure, exposure
from core.datasets_for_lunit_utils_No_MONAI import *
from core.datasets_for_No_MONAI import *

PIL.Image.MAX_IMAGE_PIXELS = 933120000 
join = os.path.join
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


from monai.data import *
from monai.metrics import *
from monai.transforms import Activations
from monai.inferers import sliding_window_inference

from core.utils import *   
from core.call_loss import *
from core.call_model import *
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
import monai
from monai.networks import one_hot

from core.metric.metrics import *




def main(info, config, args):

    activation = Activations(sigmoid=True) # softmax : odd result! ToDO : check!  
    dice_metric = DiceMetric(include_background=False, reduction='none')
    confusion_matrix = ConfusionMatrixMetric(include_background=False, reduction='none')

    os.makedirs(args.output_path, exist_ok=True)
    img_names = sorted(os.listdir(args.input_path))

    if "Thumbs.db" in img_names:
        img_names.remove("Thumbs.db")

    model = call_model(info, config).to("cuda:{0}".format(int(info["GPUS"])))
    model.load_state_dict(torch.load(os.path.join(info["LOGDIR"],args.epoch_num))["model_state_dict"])
    model.eval()

    dice_class, mr_class, fo_class = [], [], []
    real_dice, prec_class, rec_class, f1_class, h_d_class = [], [], [], [], []

    # use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:{0}".format(int(info["GPUS"])))

    roi_size = config["INPUT_SHAPE" ] # (args.input_size, args.input_size)
    sw_batch_size = 1

    # test_path = '/vast/AI_team/sukmin/Lunit_cell_test'
    # bf_norm_img_path = join(test_path, "bf_StainNorm_PIL_all")
    # af_norm_img_path = join(test_path, "af_StainNorm_PIL_all")
    # os.makedirs(bf_norm_img_path, exist_ok=True)
    # os.makedirs(af_norm_img_path, exist_ok=True)

    # inf_loader = call_dataloader_Lunit_for_inference(info, config, args.input_path)
    
    inf_loader = call_dataloader_Lunit(info, config, args.input_path, args.folder_to_inference, mode='inference')

    epoch_iterator_val = tqdm(
        inf_loader, desc="Validate (X / X Steps)", dynamic_ncols=True
    )

    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            
            t0 = time.time()

            img_name = inf_loader.dataset.img_list[step]
            step += 1

            val_inputs2, val_labels = batch
            val_inputs_i = val_inputs2.to("cuda:{0}".format(int(info["GPUS"])))
            val_labels = val_labels.to("cuda:{0}".format(int(info["GPUS"])))

            val_inputs = val_inputs_i / 255 # normalization
            val_labels = one_hot(val_labels, config["CHANNEL_OUT"])
            val_outputs = sliding_window_inference(val_inputs, config["INPUT_SHAPE"], 4 , model) # , device='cuda', sw_device='cuda')
            
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (step, len(epoch_iterator_val))
            )

            threshold=0.5
            dice_class.append(dice_metric(val_outputs>=threshold, val_labels)[0])
            confusion = confusion_matrix(val_outputs>=threshold, val_labels)[0]
            mr_class.append([
                calc_confusion_metric('fnr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
            ])
            fo_class.append([
                calc_confusion_metric('fpr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
            ])

            test_pred_out = torch.nn.functional.softmax(val_outputs, dim=1) # (B, C, H, W)
            test_pred_out = torch.argmax(test_pred_out[0], axis=0)
            label_out = torch.argmax(val_labels[0], axis=0)

            real_dice.append(dice(test_pred_out, label_out))
            prec_class.append(precision(test_pred_out, label_out))
            rec_class.append(recall(test_pred_out, label_out))
            f1_class.append(fscore(test_pred_out, label_out))




            # 검증
            test_pred_out = test_pred_out.cpu().numpy()
            label_out = label_out.cpu().numpy()

            if args.show_overlay:
                image = np.array(Image.open(join(args.input_path, img_name)))

                t1 = time.time()
                # 원본 label_out
                rst = np.zeros(image.shape, dtype=np.uint8)
                for x in range(rst.shape[0]):
                    for y in range(rst.shape[1]):
                        if label_out[x][y] == 1: # BGR
                            rst[x][y] = [255, 0, 0]   # BGR  - BC (Green)
                        elif label_out[x][y] == 2:
                            rst[x][y] = [0, 0, 255]  # TC (Red)

                alpha = 0.6
                rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)

                t2 = time.time()
                print(f'Colored finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                cv2.imwrite(join(args.output_path_for_overlap_visualize, 'vis_label' + img_name), rst)


                t1 = time.time()
                # prediction
                rst = np.zeros(image.shape, dtype=np.uint8)
                for x in range(rst.shape[0]):
                    for y in range(rst.shape[1]):
                        if test_pred_out[x][y] == 1: # BGR
                            rst[x][y] = [255, 0, 0]   # BGR  - BC (Green)
                        elif test_pred_out[x][y] == 2:
                            rst[x][y] = [0, 0, 255]  # TC (Red)

                alpha = 0.6
                rst = cv2.addWeighted(image, 1 - alpha, rst, alpha, 0.0, dtype = cv2.CV_32F)

                t2 = time.time()
                print(f'Colored finished: {img_name}; img size = {rst.shape}; costing: {t2-t1:.2f}s')
                cv2.imwrite(join(args.output_path_for_overlap_visualize, 'vis_pred' + img_name), rst)


        dice_dict, dice_val_class = calc_mean_class(info, dice_class, 'valid_dice')
        miss_dict, miss_val = calc_mean_class(info, mr_class, 'valid_miss rate')
        Val_dice = np.mean(real_dice)
        val_std = np.std(real_dice)
        Prec = np.mean(prec_class)
        Recall = np.mean(rec_class)
        F1_c = np.mean(f1_class)

        print(Val_dice)
        print(val_std)
        # print(dice_dict)

if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument('-t', dest='trainer', default='Lunit_for_paper_Cell_Only_StainNorm_BlobCell') # 'Stroma_NDM_100x_overlap')                                                             # 2022_stomach_400x
    parser.add_argument('-i', dest='input_path', default='/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/images/StainNorm/val/cell') # Task701_Lunit_Cell_only/For_inference/imagesTs  
    parser.add_argument('-o', dest='aim_path', default="r13_15k_test") # Test_200x_CacoX_b32_e45k  
    parser.add_argument('-n', dest='epoch_num', default='model_e15000.pth') # model_e06000.pth   # model_best_e09900.pth  # class_model_best_e10001.pth
    parser.add_argument('-f_gt', dest='folder_to_inference', default='/vast/AI_team/sukmin/datasets/Lunit_Challenge_for_paper/labels/val/BlobCell')
    # /vast/AI_team/sukmin/datasets/Lunit_Challenge/Task701_Lunit_Cell_only/For_inference_Split/imagesTs
    # /vast/AI_team/sukmin/datasets/Lunit_Challenge/Task701_Lunit_Cell_only/For_inference_Split/imagesTs
    # /vast/AI_team/sukmin/datasets/Lunit_Challenge/Task749_Lunit_Cell_StainNorm_BlobCell_JustCell_Split/labelsTs
    parser.add_argument('-show_overlay', required=False, default=False, action="store_true", help='save segmentation overlay')
    parser.add_argument('-show_prediction', required=False, default=False, action="store_true", help='save segmentation output')
    parser.add_argument('-make_WhiteSpace', required=False, default=False, action="store_true", help='stroma become whitespace and save it')
    parser.add_argument('-for_sudo_labeling', required=False, default=False, action="store_true", help='inference for labeling')

    args = parser.parse_args()

    args.output_path = join('/vast/AI_team/sukmin/Results_Test_Lunit_Challenge_for_paper', 'like_wandb', 'val', 'r12', args.aim_path)
    args.output_path_for_overlap_visualize = join('/vast/AI_team/sukmin/Results_Test_Lunit_Challenge_for_paper/val/overlap_vis', args.aim_path)
    args.output_path_for_visualize = join('/vast/AI_team/sukmin/Results_Test_Lunit_Challenge_for_paper/result_vis', args.aim_path)
    # args.output_path_for_WhiteSpace_rst = join('/vast/AI_team/sukmin/Results_WhiteSpace_data', args.aim_path)
    # args.output_path_for_sudo_label = join('/vast/AI_team/sukmin/Results_for_sudo_label', args.aim_path)

    os.makedirs(args.output_path, exist_ok=True)
    if args.show_overlay:
        os.makedirs(args.output_path_for_overlap_visualize, exist_ok=True)
    if args.show_prediction:
        os.makedirs(args.output_path_for_visualize, exist_ok=True)

    if args.make_WhiteSpace:    
        os.makedirs(args.output_path_for_WhiteSpace_rst, exist_ok=True)

    info, config = call_config(args.trainer)
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, args)
# %%
