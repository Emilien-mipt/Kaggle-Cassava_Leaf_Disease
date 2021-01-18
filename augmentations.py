from albumentations import (CenterCrop, Compose, HorizontalFlip, Normalize,
                            RandomResizedCrop, Resize, Transpose, VerticalFlip)
from albumentations.pytorch import ToTensorV2

from config import CFG


# ====================================================
# Transforms
# ====================================================
def get_transforms(*, data):

    if data == "train":
        return Compose(
            [
                Resize(CFG.size, CFG.size, p=1.0),
                # RandomResizedCrop(CFG.size, CFG.size, p=1.),
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
                Resize(CFG.size, CFG.size),
                # CenterCrop(CFG.size, CFG.size, p=1.),
                Normalize(
                    mean=CFG.MEAN,
                    std=CFG.STD,
                ),
                ToTensorV2(),
            ]
        )
