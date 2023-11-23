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

        self.unet = DeepLabV3Plus(
            encoder_name="se_resnext101_32x4d",  
            encoder_weights="imagenet",
            in_channels=3,
            classes=3,
        )
        
        self.load_checkpoint()
        self.unet = self.unet.to(self.device)
        self.unet.eval()
        use_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda" if use_cuda else "cpu")

    def load_checkpoint(self):
        """Loading the trained weights to be used for validation"""
        _curr_path = os.path.split(__file__)[0]
        # _path_to_checkpoint = os.path.join(_curr_path, "checkpoints/Has_Second.pth")
        _path_to_checkpoint = "/vast/AI_team/sukmin/Results_monai_Lunit_Challenge_for_paper/Cell_Only_StainNorm_BlobCell_No_MONAI/model_e27000.pth" # model_e15000.pth# model_best_800.pth
        state_dict = torch.load(_path_to_checkpoint, map_location=torch.device('cpu'))
        self.unet.load_state_dict(state_dict['model_state_dict'])
        print("Weights were successfully loaded!")

        
    def prepare_input(self, cell_patch):
        """This function prepares the cell patch array to be forwarded by
        the model

        Parameters
        ----------
        cell_patch: np.ndarray[uint8]
            Cell patch with shape [1024, 1024, 3] with values from 0 - 255

        Returns
        -------
            torch.tensor of shape [1, 3, 1024, 1024] where the first axis is the batch
            dimension
        """

        # Adapt StainNorm
        # cell_data_norm = normalizeStaining(cell_patch)
        cell_data_norm = cell_patch
        cell_data_norm = cell_data_norm/255
        cell_data_norm = torch.from_numpy(np.expand_dims(cell_data_norm, 0)).permute(0,3,1,2).type(torch.FloatTensor).to(self.device)
        return cell_data_norm
        
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

    def post_process(self, logits):
        """This function applies some post processing to the
        output logits
        
        Parameters
        ----------
        logits: torch.tensor
            Outputs of U-Net

        Returns
        -------
            torch.tensor after post processing the logits
        """
        if self.resize_to is not None:
            logits = F.interpolate(logits, size=SAMPLE_SHAPE[:2],
                mode='bilinear', align_corners=False
            )
        return torch.softmax(logits, dim=1)

    def __call__(self, cell_patch, tissue_patch, pair_id):
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
        cell_patch = self.prepare_input(cell_patch)
        logits = self.unet(cell_patch)
        # logits = sliding_window_inference(cell_patch, self.input_shape, sw_batch_size=1, predictor=self.unet)
        heatmap = self.post_process(logits)
        return self.find_cells(heatmap)
