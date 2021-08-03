import argparse
import os
import time

import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    ReduceLROnPlateau,
)
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch_lr_finder import LRFinder

from augmentations import get_transforms
from config import CFG
from model import CustomModel
from train import train_fn, valid_fn
from train_test_dataset import TrainDataset
from utils.loss_functions import get_criterion
from utils.utils import get_score, init_logger, save_batch, seed_torch, weight_class


def main():
    parser = argparse.ArgumentParser(description="Parse the argument to define the train log dir name")
    parser.add_argument(
        "--logdir_name",
        type=str,
        help="Name of the dir to save train logs",
    )
    parser.add_argument(
        "--save_batch_fig",
        action="store_true",
        help="Whether to save a sample of a batch or not",
    )
    parser.add_argument(
        "--find_lr",
        action="store_true",
        help="Whether to find optimal learning rate or not",
    )

    args = parser.parse_args()
    log_dir_name = args.logdir_name
    save_single_batch = args.save_batch_fig
    find_lr = args.find_lr

    # Path to log
    logger_path = os.path.join(CFG.OUTPUT_DIR, log_dir_name)

    # Create dir for saving logs and weights
    print(f"Creating dir {log_dir_name} for saving logs")
    os.makedirs(os.path.join(logger_path, "weights"))
    print(f"Dir {log_dir_name} has been created!")

    # Define logger to save train logs
    LOGGER = init_logger(os.path.join(logger_path, "train.log"))
    # Write to tensorboard
    tb = SummaryWriter(logger_path)

    # Set seed
    seed_torch(seed=CFG.seed)

    LOGGER.info("Reading data...")
    train_df = pd.read_csv("./data/cassava-leaf-disease-classification/train.csv")

    CLASS_NAMES = ["CBB", "CBSD", "CGM", "CMD", "Healthy"]
    # weight_list = weight_class(train_df)
    # LOGGER.info(f"Weight list for classes: {weight_list}")

    LOGGER.info("Splitting data for training and validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        train_df.loc[:, train_df.columns != "label"],
        train_df["label"],
        test_size=0.2,
        random_state=CFG.seed,
        shuffle=True,
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

    device = torch.device(f"cuda:{CFG.GPU_ID}")

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

    # Show batch to see the effect of augmentations
    if save_single_batch:
        LOGGER.info("Creating dir to save samples of a batch...")
        path_to_figs = os.path.join(logger_path, "batch_figs")
        os.makedirs(path_to_figs)
        LOGGER.info("Saving figures of a single batch...")
        save_batch(train_loader, CLASS_NAMES, path_to_figs, CFG)
        LOGGER.info("Figures have been saved!")

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG.model_name, pretrained=True)
    model.to(device)

    LOGGER.info(f"Model name {CFG.model_name}")
    LOGGER.info(f"Batch size {CFG.batch_size}")
    LOGGER.info(f"Input size {CFG.size}")

    # optimizer = Adam(model.parameters(), lr=CFG.lr)
    optimizer = SGD(model.parameters(), lr=CFG.lr, momentum=CFG.momentum, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, base_lr=CFG.min_lr, max_lr=CFG.lr, mode="triangular2", step_size_up=2138
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    #    optimizer, T_0=1, T_mult=2, eta_min=0.001, verbose=True
    # )
    # ====================================================
    # loop
    # ====================================================
    criterion = get_criterion()
    # criterion = nn.CrossEntropyLoss()
    LOGGER.info(f"Select {CFG.criterion} criterion")

    if find_lr:
        print("Finding oprimal learning rate...")
        # Add this line before running `LRFinder`
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda", log_path=logger_path)
        lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
        lr_finder.plot()  # to inspect the loss-learning rate graph
        lr_finder.reset()  # to reset the model and optimizer to their initial state
        print("Optimal learning rate has been found!")

    best_epoch = 0
    best_acc_score = 0.0
    best_f1_score = 0.0

    count_bad_epochs = 0  # Count epochs that don't improve the score

    if CFG.MIXED_PREC:
        LOGGER.info("Enabling mixed precision for training...")

    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(CFG.epochs):

        start_time = time.time()

        # train
        avg_train_loss, train_acc = train_fn(
            train_loader, model, criterion, optimizer, scaler, epoch, device, scheduler
        )

        # eval
        avg_val_loss, val_preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_fold[CFG.target_col].values

        # scoring on validation set
        val_acc_score = get_score(valid_labels, val_preds.argmax(1), metric="accuracy")
        val_f1_score = get_score(valid_labels, val_preds.argmax(1), metric="f1_score")

        cur_lr = optimizer.param_groups[0]["lr"]
        # scheduler.step(val_acc_score)
        LOGGER.info(f"Current learning rate: {cur_lr}")

        tb.add_scalar("Learning rate", cur_lr, epoch + 1)
        tb.add_scalar("Train Loss", avg_train_loss, epoch + 1)
        tb.add_scalar("Train accuracy", train_acc, epoch + 1)
        tb.add_scalar("Val Loss", avg_val_loss, epoch + 1)
        tb.add_scalar("Val accuracy score", val_acc_score, epoch + 1)
        tb.add_scalar("Val f1 score", val_f1_score, epoch + 1)

        elapsed = time.time() - start_time

        LOGGER.info(
            f"Epoch {epoch+1} - avg_train_loss: {avg_train_loss:.4f} \
            avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s"
        )
        LOGGER.info(f"Epoch {epoch+1} - Accuracy: {val_acc_score} - F1-score {val_f1_score}")

        best_acc_bool = False
        best_f1_bool = False

        # Update best score
        if val_acc_score >= best_acc_score:
            best_acc_score = val_acc_score
            best_acc_bool = True

        if val_f1_score >= best_f1_score:
            best_f1_score = val_f1_score
            best_f1_bool = True

        if best_acc_bool and best_f1_bool:
            LOGGER.info(
                f"Epoch {epoch+1} - Save Best Accuracy: {best_acc_score:.4f} - \
                Save Best F1-score: {best_f1_score:.4f} Model"
            )
            torch.save(
                {"model": model.state_dict(), "preds": val_preds},
                os.path.join(
                    logger_path,
                    "weights",
                    "best.pth",
                ),
            )
            best_epoch = epoch + 1
            count_bad_epochs = 0
        else:
            count_bad_epochs += 1
        print(count_bad_epochs)
        LOGGER.info(f"Number of bad epochs {count_bad_epochs}")
        # Early stopping
        if count_bad_epochs > CFG.early_stopping:
            LOGGER.info(f"Stop the training, since the score has not improved for {CFG.early_stopping} epochs!")
            torch.save(
                {"model": model.state_dict(), "preds": val_preds},
                os.path.join(
                    logger_path,
                    "weights",
                    f"{CFG.model_name}_epoch{epoch+1}_last.pth",
                ),
            )
            break
        elif epoch + 1 == CFG.epochs:
            LOGGER.info(f"Reached the final {epoch+1} epoch!")
            torch.save(
                {"model": model.state_dict(), "preds": val_preds},
                os.path.join(
                    logger_path,
                    "weights",
                    f"{CFG.model_name}_epoch{epoch+1}_final.pth",
                ),
            )

    LOGGER.info(
        f"AFTER TRAINING: Epoch {best_epoch}: Best Accuracy: {best_acc_score:.4f} - \
                Best F1-score: {best_f1_score:.4f}"
    )
    tb.close()


if __name__ == "__main__":
    main()
