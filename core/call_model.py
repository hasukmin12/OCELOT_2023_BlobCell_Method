import os, glob
import torch
from torch import optim
from monai.data import *
import segmentation_models_pytorch as smp

def call_model(info, config):
    model = None

    if config["MODEL_NAME"] in ['DL_se_resnext101']:
        model = smp.DeepLabV3Plus(
            encoder_name="se_resnext101_32x4d", 
            encoder_weights="imagenet",
            in_channels=config["CHANNEL_IN"],
            classes=config["CHANNEL_OUT"],
        )


    assert model is not None, 'Model Error!'    
    return model


def call_optimizer(config, model):
    if config["OPTIM_NAME"] in ['SGD', 'sgd']:
        return optim.SGD(model.parameters(), lr=config["LR_INIT"], momentum=config["MOMENTUM"])
    elif config["OPTIM_NAME"] in ['ADAM', 'adam', 'Adam']:
        return optim.Adam(model.parameters(), lr=config["LR_INIT"])
    elif config["OPTIM_NAME"] in ['ADAMW', 'adamw', 'AdamW', 'Adamw']:
        return optim.AdamW(model.parameters(), lr=config["LR_INIT"], weight_decay=config["LR_DECAY"])
    elif config["OPTIM_NAME"] in ['ADAGRAD', 'adagrad', 'AdaGrad']:
        return optim.Adagrad(model.parameters(), lr=config["LR_INIT"], lr_decay=config["LR_DECAY"])
    else:
        return None
