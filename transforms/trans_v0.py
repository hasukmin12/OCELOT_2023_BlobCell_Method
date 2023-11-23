from monai.transforms import *
import math

def call_transforms(config):
    ## train transforms
    train_transforms = [LoadImaged(keys=["image", "label"])]    
    val_transforms   = [LoadImaged(keys=["image", "label"])]
    if config["SPACING"] is not None:
        spacing = Spacingd(
            keys=["image","label"],
            pixdim=config["SPACING"],
            mode=["bilinear","nearest"]
        )
        train_transforms += [spacing]
        val_transforms += [spacing]

    train_transforms += [
        EnsureChannelFirstd(keys=["image", "label"]),
        AsDiscreted(keys=["label"], to_onehot=config["CHANNEL_OUT"]), 
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[0],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[1],
            prob=0.10,
        ),
        RandFlipd(
            keys=["image", "label"],
            spatial_axis=[2],
            prob=0.10,
        ),        
        RandStdShiftIntensityd(
            keys=["image"],
            prob=0.30,
            factors=(-10,10),
        ),
        RandHistogramShiftd(
            keys=["image"],
            num_control_points=10,
            prob=0.30,
        ),
        RandAdjustContrastd(
            keys=["image"],
            prob=0.30,
            gamma=(0.5,2.0),
        ),
        RandZoomd(
            keys=["image","label"],
            prob=0.30,
            min_zoom=0.8,
            max_zoom=1.2,
            keep_size=True,
        ),
        RandRotated(
            keys=["image","label"],
            range_x=0.4,
            range_y=0.4,
            range_z=0.4,
            prob=0.30,
        ),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config["CONTRAST"][0],
            a_max=config["CONTRAST"][1],
            b_min=0, b_max=1, clip=True,
        ),
        RandCropByPosNegLabeld(
            keys=["image","label"], 
            label_key="label",
            pos=1,
            neg=1,
            num_samples=config["SAMPLES"],
            spatial_size=config["INPUT_SHAPE"], 
        ),
        EnsureTyped(keys=["image", "label"]),      
    ]
     
    ## validation transforms
    val_transforms += [
        EnsureChannelFirstd(keys=["image", "label"]),
        AsDiscreted(keys=["label"], to_onehot=config["CHANNEL_OUT"]),        
        ScaleIntensityRanged(
            keys=["image"],
            a_min=config["CONTRAST"][0],
            a_max=config["CONTRAST"][1],
            b_min=0, b_max=1, clip=True,
        ),
        EnsureTyped(keys=["image", "label"]),
    ]
    
    if config["FAST"]:
        train_transforms += [ToDeviced(keys=["image", "label"], device="cuda:0")]
        val_transforms += [ToDeviced(keys=["image", "label"], device="cuda:0")]
    return Compose(train_transforms), Compose(val_transforms)