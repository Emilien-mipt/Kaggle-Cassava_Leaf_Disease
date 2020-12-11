import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from augmentations import get_transforms
from config import CFG
from data_processing import TestDataset, TrainDataset
from model import CustomModel
from train import train_fn, valid_fn
from utils.utils import get_score, init_logger, seed_torch


def main():
    LOGGER = init_logger("train.log")
    tb = SummaryWriter(CFG.OUTPUT_DIR)
    seed_torch(seed=CFG.seed)

    LOGGER.info("Reading data...")
    train_df = pd.read_csv("./data/cassava-leaf-disease-classification/train.csv")

    LOGGER.info("Splitting data for training and validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_df.loc[:, train_df.columns != "label"],
        train_df["label"],
        test_size=0.2,
        random_state=CFG.seed,
        shuffle = True,
        stratify=train_df["label"],
    )

    train_fold = pd.concat([X_train, y_train], axis=1)
    LOGGER.info("train shape: ")
    LOGGER.info(train_fold.shape)
    valid_fold = pd.concat([X_val, y_val], axis=1)
    LOGGER.info("valid shape: ")
    LOGGER.info(valid_fold.shape)

    LOGGER.info("train fold: ")
    LOGGER.info(train_fold.groupby([CFG.target_col]).size())
    LOGGER.info("validation fold: ")
    LOGGER.info(valid_fold.groupby([CFG.target_col]).size())

    # ====================================================
    # Form dataloaders
    # ====================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = TrainDataset(train_fold, transform=get_transforms(data="train"))
    valid_dataset = TrainDataset(valid_fold, transform=get_transforms(data="valid"))

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG.model_name, pretrained=True)
    model.to(device)

    optimizer = Adam(model.parameters(), lr=CFG.lr)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()

    best_score = 0.0
    best_loss = np.inf

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_train_loss, train_score = train_fn(
            train_loader, model, criterion, optimizer, epoch, device
        )

        # eval
        avg_val_loss, val_preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_fold[CFG.target_col].values

        # scoring on validation set
        val_score = get_score(valid_labels, val_preds.argmax(1))

        tb.add_scalar("Train Loss", avg_train_loss, epoch+1)
        tb.add_scalar("Train accuracy", train_score, epoch+1)
        tb.add_scalar("Val Loss", avg_val_loss, epoch+1)
        tb.add_scalar("Val accuracy", val_score, epoch+1)

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Accuracy: {val_score}")

        if val_score > best_score:
            best_score = val_score
            LOGGER.info(f"Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model")
            torch.save(
                {"model": model.state_dict(), "preds": val_preds},
                CFG.OUTPUT_DIR + f"{CFG.model_name}_epoch{epoch}_best.pth",
            )
    tb.close()

if __name__ == "__main__":
    main()
