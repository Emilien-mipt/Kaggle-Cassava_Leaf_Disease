"""
Calculation Of standard deviation and mean (per channel) over all images of the image dataset
"""
import cv2
import pandas as pd
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_transforms
from config import CFG
from train_test_dataset import TrainDataset
from utils.utils import seed_torch


def get_mean_std_opencv(train_df):
    sum_mean = 0.0
    sum_std = 0.0
    file_name_list = train_df["image_id"].values
    for file_name in file_name_list:
        print("Processing file ", file_name)
        file_path = f"{CFG.TRAIN_PATH}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (CFG.size, CFG.size))
        mean, std = cv2.meanStdDev(resized_image)
        sum_mean += mean
        sum_std += std
    avg_mean = sum_mean / len(file_name_list)
    avg_std = sum_std / len(file_name_list)
    return avg_mean / 255.0, avg_std / 255.0


def get_mean_std(loader):
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for data, _ in tqdm(loader):
        data = torch.Tensor.float(data)
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        data = data
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples
    mean /= nb_samples
    std /= nb_samples
    return mean / 255.0, std / 255


def main():
    # Set seed
    seed_torch(seed=CFG.seed)

    print("Reading data...")
    train_df = pd.read_csv("./data/cassava-leaf-disease-classification/train.csv")
    print("Splitting data for training and validation...")

    X_train, X_val, y_train, y_val = train_test_split(
        train_df.loc[:, train_df.columns != "label"],
        train_df["label"],
        test_size=0.2,
        random_state=CFG.seed,
        shuffle=True,
        stratify=train_df["label"],
    )

    print(X_train)

    train_fold = pd.concat([X_train, y_train], axis=1)
    valid_fold = pd.concat([X_val, y_val], axis=1)
    print("train shape: ")
    print(train_fold.shape)
    print("valid shape: ")
    print(valid_fold.shape)

    # ====================================================
    # Form dataloaders
    # ====================================================

    train_dataset = TrainDataset(train_fold, transform=get_transforms(data="train"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    MEAN, STD = get_mean_std(train_loader)
    MEAN_2, STD_2 = get_mean_std_opencv(train_df)

    print("MEAN: ", MEAN, "STD: ", STD)
    print("MEAN: ", MEAN_2, "STD: ", STD_2)


if __name__ == "__main__":
    main()
