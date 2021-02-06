from albumentations import (
    CenterCrop,
    CoarseDropout,
    Compose,
    Cutout,
    HorizontalFlip,
    Normalize,
    RandomBrightnessContrast,
    RandomResizedCrop,
    RandomRotate90,
    Resize,
    ShiftScaleRotate,
    Transpose,
    VerticalFlip,
)
from albumentations.pytorch import ToTensorV2

from config import CFG


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):

    if data == "train":
        return Compose(
            [
                # Resize(CFG.size, CFG.size, p=1.0),
                RandomResizedCrop(CFG.size, CFG.size, p=1.0),
                # RandomBrightnessContrast(
                #    p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=False
                # ),
                # ShiftScaleRotate(
                #    p=0.5,
                #    shift_limit=(-0.3, 0.3),
                #    scale_limit=(-0.1, 0.1),
                #    rotate_limit=(-180, 180),
                #    interpolation=4,
                #    border_mode=4,
                # ),
                # CoarseDropout(
                #    p=0.5, max_holes=12, max_height=12, max_width=12, min_holes=8, min_height=8, min_width=8
                # ),
                # Cutout(p=0.5, num_holes=12),
                Transpose(p=0.5),
                HorizontalFlip(p=0.5),
                VerticalFlip(p=0.5),
                Normalize(
                    mean=CFG.MEAN,
                    std=CFG.STD,
                ),
                ToTensorV2(),
            ]
        )

    elif data == "valid":
        return Compose(
            [
                # Resize(CFG.size, CFG.size),
                CenterCrop(CFG.size, CFG.size, p=1.0),
                Normalize(
                    mean=CFG.MEAN,
                    std=CFG.STD,
                ),
                ToTensorV2(),
            ]
        )
