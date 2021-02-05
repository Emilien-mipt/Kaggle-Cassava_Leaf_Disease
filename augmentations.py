from albumentations import (
    CenterCrop,
    Compose,
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
                RandomBrightnessContrast(p=0.5, brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), brightness_by_max=False),
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
