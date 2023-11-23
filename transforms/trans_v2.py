from monai.transforms import *
import numpy as np
import math

# for semi-supervised learning

def call_transforms_for_semi(config):
    train_transforms = [
            LoadImaged(
                keys=["image", "label"], dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["image"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["image"], allow_missing_keys=True
            ),  # Do not scale label
            SpatialPadd(keys=["image", "label"], spatial_size=config["INPUT_SHAPE"]),
            RandSpatialCropd(
                keys=["image", "label"], roi_size=config["INPUT_SHAPE"], random_size=False
            ),

            # RandCropByPosNegLabeld(
            #     keys=["image", "label"], spatial_size=config["INPUT_SHAPE"], label_key='label',
            #     pos=1, neg=0.1, num_samples=2
            # ),


            RandAxisFlipd(keys=["image", "label"], prob=0.5),
            RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["image"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["image"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["image"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["image", "label"],
                prob=0.15,
                min_zoom=0.8,
                max_zoom=1.5,
                mode=["area", "nearest"],
            ),
            EnsureTyped(keys=["image", "label"]),
        ]
    

    val_transforms = [
            LoadImaged(keys=["image", "label"], dtype=np.uint8),
            AddChanneld(keys=["label"], allow_missing_keys=True),
            AsChannelFirstd(keys=["image"], channel_dim=-1, allow_missing_keys=True),
            ScaleIntensityd(keys=["image"], allow_missing_keys=True),
            # AsDiscreted(keys=['label'], to_onehot=3),
            EnsureTyped(keys=["image", "label"]),
        ]

    unlabel_transforms = [
            LoadImaged(
                keys=["image"], dtype=np.uint8
            ),  # image three channels (H, W, 3); label: (H, W)
            # AddChanneld(keys=["label"], allow_missing_keys=True),  # label: (1, H, W)
            AsChannelFirstd(
                keys=["image"], channel_dim=-1, allow_missing_keys=True
            ),  # image: (3, H, W)
            ScaleIntensityd(
                keys=["image"], allow_missing_keys=True
            ),  # Do not scale label
            SpatialPadd(keys=["image"], spatial_size=config["INPUT_SHAPE"]),
            RandSpatialCropd(
                keys=["image"], roi_size=config["INPUT_SHAPE"], random_size=False
            ),
            # RandCropByPosNegLabeld(
            #     keys=["image"], spatial_size=config["INPUT_SHAPE"], label_key=["image"]
            #     pos=1, neg=0.1, num_samples=4
            # ),
            RandAxisFlipd(keys=["image"], prob=0.5),
            RandRotate90d(keys=["image"], prob=0.5, spatial_axes=[0, 1]),
            # # intensity transform
            RandGaussianNoised(keys=["image"], prob=0.25, mean=0, std=0.1),
            RandAdjustContrastd(keys=["image"], prob=0.25, gamma=(1, 2)),
            RandGaussianSmoothd(keys=["image"], prob=0.25, sigma_x=(1, 2)),
            RandHistogramShiftd(keys=["image"], prob=0.25, num_control_points=3),
            RandZoomd(
                keys=["image"],
                prob=0.15,
                min_zoom=0.8,
                max_zoom=1.5,
                mode=["area"],
            ),
            EnsureTyped(keys=["image"]),
        ]

    if config["FAST"]:
        train_transforms += [ToDeviced(keys=["image", "label"], device="cuda:0")]
        val_transforms += [ToDeviced(keys=["image", "label"], device="cuda:0")]

    return Compose(train_transforms), Compose(val_transforms), Compose(unlabel_transforms)