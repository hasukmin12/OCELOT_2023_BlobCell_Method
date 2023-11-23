import os
import numpy as np
from skimage import feature
import cv2
from util.constants import SAMPLE_SHAPE

import torch
import torch.nn as nn
import torch.nn.functional as F

# import segmentation_models_pytorch as smp
from user.model.deeplabv3 import DeepLabV3Plus
from monai.inferers import sliding_window_inference
from util.StainNormalize import normalizeStaining
from util.gcio import read_json


class PytorchUnetCellModel():
    """
    U-NET model for cell detection implemented with the Pytorch library

    NOTE: this model does not utilize the tissue patch but rather
    only the cell patch.

    Parameters
    ----------
    metadata: Dict
        Dataset metadata in case you wish to compute statistics

    """
    def __init__(self, metadata):
        self.device = torch.device('cuda:0')
        self.metadata = metadata
        # RGB images and 2 class prediction
        self.resize_to = None
        self.n_classes =  3 # Two cell classes and background
        self.input_shape = (1024, 1024)

        # Main Model
        self.main_model = DeepLabV3Plus(
            encoder_name="se_resnext101_32x4d",  
            encoder_weights="imagenet",
            in_channels=4,
            classes=3,
        )


        # Load models
        _curr_path = os.path.split(__file__)[0]                                                                                                                      # model_e15750_dice0.766632584262057.pth
        # self.main_model_path = "/vast/AI_team/sukmin/Results_monai_Lunit_Challenge/Cell_r14_2_concat_StainNorm_No_MONAI/model_e15750_dice0.570865699194163.pth"    # model_e19250_dice0.7778223295458635.pth"
        # _path_to_checkpoint = os.path.join(self.main_model_path)                                                                                                   # model_e22750_dice0.779868354443114.pth"
        _path_to_checkpoint = os.path.join(_curr_path, "/vast/AI_team/sukmin/Results_monai_Lunit_Challenge_for_paper/Cell_r14_2_concat_StainNorm_No_MONAI/model_e15750_dice0.570865699194163.pth")
        # state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))
        state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cuda:0'))
        self.main_model.load_state_dict(state_dict['model_state_dict'])
        self.main_model = self.main_model.to(self.device)
        self.main_model.eval()

        use_cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0')

    def find_cells(self, heatmap):
        """This function detects the cells in the output heatmap

        Parameters
        ----------
        heatmap: torch.tensor
            output heatmap of the model,  shape: [1, 3, 1024, 1024]

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """
        arr = heatmap[0,:,:,:].cpu().detach().numpy()
        # arr = np.transpose(arr, (1, 2, 0)) # CHW -> HWC

        bg, pred_wo_bg = np.split(arr, (1,), axis=0) # Background and non-background channels
        bg = np.squeeze(bg, axis=0)
        obj = 1.0 - bg

        arr = cv2.GaussianBlur(obj, (0, 0), sigmaX=3)
        peaks = feature.peak_local_max(
            arr, min_distance=3, exclude_border=0, threshold_abs=0.0
        ) # List[y, x]

        maxval = np.max(pred_wo_bg, axis=0)
        maxcls_0 = np.argmax(pred_wo_bg, axis=0)

        # Filter out peaks if background score dominates
        peaks = np.array([peak for peak in peaks if bg[peak[0], peak[1]] < maxval[peak[0], peak[1]]])
        if len(peaks) == 0:
            return []

        # Get score and class of the peaks
        scores = maxval[peaks[:, 0], peaks[:, 1]]
        peak_class = maxcls_0[peaks[:, 0], peaks[:, 1]]

        predicted_cells = [(x, y, c + 1, float(s)) for [y, x], c, s in zip(peaks, peak_class, scores)]

        return predicted_cells



    def __call__(self, cell_patch, tissue_patch, blobcell_patch, pair_id):
        """This function detects the cells in the cell patch using Pytorch U-Net.

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255
        tissue_patch: np.ndarray[uint8] 
            Tissue patch with shape [1024, 1024, 3] with values from 0 - 255
        pair_id: str
            Identification number of the patch pair

        Returns
        -------
            List[tuple]: for each predicted cell we provide the tuple (x, y, cls, score)
        """

        # StainNormalization
        # Stained_cell = normalizeStaining(cell_patch)/255
        # Stained_tissue = normalizeStaining(tissue_patch)/255
        cell_patch = cell_patch/255
        
        # To tensor
        test_tensor = torch.from_numpy(np.expand_dims(cell_patch, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(self.device)
        t_tensor = torch.from_numpy(np.expand_dims(tissue_patch, 0)).unsqueeze(0).type(torch.FloatTensor).to(self.device)
        b_tensor = torch.from_numpy(np.expand_dims(blobcell_patch, 0)).unsqueeze(0).type(torch.FloatTensor).to(self.device)



        # concat three images
        test_tensor = torch.cat((test_tensor, t_tensor), dim=1)


        # inference Final model
        test_pred_out = self.main_model(test_tensor)
        # test_pred_out = sliding_window_inference(test_tensor, roi_size, sw_batch_size, model)[0]

        test_pred_out = torch.nn.functional.softmax(test_pred_out, dim=1) # (B, C, H, W)
        # test_pred_npy = test_pred_out[0,1].cpu().numpy()
        # test_pred_npy = test_pred_out[0].cpu().numpy()
        # rst = np.argmax(test_pred_npy, axis=0)


        return self.find_cells(test_pred_out)
