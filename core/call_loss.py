import torch.nn as nn
from monai.losses import * # DiceLoss, DiceCELoss, GeneralizedDiceLoss, DiceFocalLoss
from core.loss.losses import * # DiceFocalLoss_Portion, DiceCELoss_Portion
from core.loss.AdaptiveWing import AWing
from core.loss.Hausdorff import HausdorffDTLoss, HausdorffERLoss
from core.loss.Boundary import *
from core.loss.MONAI_losses import DiceCELoss2

class call_loss(nn.Module):  
    def __init__(self, 
                loss_mode, 
                include_background=False, 
                sigmoid=False, 
                softmax=False, 
                y_onehot=False,
                config=None):
        super().__init__()
        self.CE_Loss = CE_Loss(
            include_background=include_background, 
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            )
        self.Dice = DiceLoss(
            include_background=include_background, 
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            )
        self.Focal = FocalLoss(
            )
        self.GDice = GeneralizedDiceLoss(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            )
        self.DiceCE = DiceCELoss2(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            ) 
        self.DiceFocal = DiceFocalLoss(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            )
        self.DiceCE_Portion = DiceCELoss_Portion(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
            )
        self.DiceFocal_Portion = DiceFocalLoss_Portion(
            include_background=include_background,
            sigmoid=sigmoid, 
            softmax=softmax, 
            to_onehot_y=y_onehot
        )
        # self.Wassert_Dice = GeneralizedWassersteinDiceLoss(
        #     include_background=include_background,
        #     sigmoid=sigmoid, 
        #     softmax=softmax, 
        #     to_onehot_y=y_onehot
        # )



        self.HausdorffDTLoss = HausdorffDTLoss(alpha = 2.0) # you can modify the parameter!
        self.HausdorffERLoss = HausdorffERLoss(alpha = 2.0, erosions = 10) # you can modify the parameters!
        self.BoundaryLoss = BDLoss()
        # self.DC_and_BD_loss = DC_and_BD_loss() # you can modify the parameter!
        self.DistBinaryDiceLoss = DistBinaryDiceLoss()

        if loss_mode in ['Adaptive_Wing', 'adaptive_wing', 'AdaptiveWing', 'Awing']:
            self.AdaptiveWing = AWing(
                alpha=config["Awing_alpha"],
                omega=config["Awing_omega"],
                epsilon=config["Awing_epsilon"],
                theta=config["Awing_theta"],
                activation=config["Awing_activation"],
            )
        self.LOSS_NAME = loss_mode

    def forward(self, pred, target):
        if self.LOSS_NAME in ['ce', 'CE', 'Ce']:
            return self.CE_Loss(pred, target)
        elif self.LOSS_NAME in ['dice', 'Dice', 'DICE']:
            return self.Dice(pred, target)
        elif self.LOSS_NAME in ['focal', 'Focal']:
            return self.Focal(pred, target)
        elif self.LOSS_NAME in ['gdice', 'GDice', 'Gdice', 'gen_dice']:
            return self.GDice(pred, target)
        elif self.LOSS_NAME in ['Dice+CrossEntropy', 'DiceCE', 'dice+ce', 'dice_ce']:
            return self.DiceCE(pred, target)
        elif self.LOSS_NAME in ['DiceFocal', 'dicefocal', 'Dice+Focal', 'dice+focal', 'dice_focal']:
            return self.DiceFocal(pred, target)
        elif self.LOSS_NAME in ['DiceCE_Portion', 'diceceportion', 'Dice+Ce+Portion', 'dice+ce+portion', 'dice_ce_portion']:
            return self.DiceCE_Portion(pred, target)
        elif self.LOSS_NAME in ['DiceFocal_Portion', 'dicefocalportion', 'Dice+Focal+Portion', 'dice+focal+portion', 'dice_focal_portion']:
            return self.DiceFocal_Portion(pred, target)
        elif self.LOSS_NAME in ['Adaptive_Wing', 'adaptive_wing', 'AdaptiveWing', 'Awing']:
            return self.AdaptiveWing(pred, target)
        elif self.LOSS_NAME in ['HausdorffDTLoss', 'HD_DT', 'HD_distance', 'hd_distance']:
            return self.HausdorffDTLoss(pred, target)
        elif self.LOSS_NAME in ['HausdorffERLoss', 'HD_ER', 'HD_erosion', 'hd_erosion']:
            return self.HausdorffERLoss(pred, target)
        elif self.LOSS_NAME in ['boundary', 'Boundary']:
            return self.BoundaryLoss(pred, target)
        elif self.LOSS_NAME in ['boundary_dice', 'BoundaryDice']:
            return self.DC_and_BD_loss(pred, target)
        elif self.LOSS_NAME in ['distance_dice', 'DistanceDice']:
            return self.DistBinaryDiceLoss(pred, target)
        # elif self.LOSS_NAME in ['Wassert_Dice', 'W_dice']:
        #     return self. Wassert_Dice(pred, target)