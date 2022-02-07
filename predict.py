import argparse

import pandas as pd
import torch
from torch.utils.data import DataLoader

from augmentations import get_transforms
from config import CFG
from inference import inference
from model import CustomModel
from train_test_dataset import TestDataset


def predict(test_fold, state, device):
    model = CustomModel(CFG.model_name, pretrained=False, num_classes=CFG.target_size)
    test_dataset = TestDataset(test_fold, transform=get_transforms(data="valid"))
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    predictions = inference(model, state, test_loader, device)
    # submission
    test_fold["label"] = predictions.argmax(1)
    test_fold[["image_id", "label"]].to_csv("submission.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description="Parse the arguments to define the dict state for the model")
    parser.add_argument(
        "--state",
        type=str,
        help="Model state, which will be used to load the model weights",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_fold = pd.read_csv("./data/cassava-leaf-disease-classification/sample_submission.csv")
    state = args.state
    predict(test_fold, state, device)


if "__name__" == "__main__":
    main()
