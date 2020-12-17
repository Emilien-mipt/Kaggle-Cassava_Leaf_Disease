import time

import numpy as np
import torch
from tqdm import tqdm

from config import CFG
from utils.utils import AverageMeter, timeSince, get_score


def train_fn(train_loader, model, criterion, optimizer, epoch, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracy = AverageMeter()
    # switch to train mode
    model.train()
    start = end = time.time()

    # Iterate over dataloader
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.to(device)
        labels = labels.to(device)

        batch_size = labels.size(0)
        y_preds = model(images)
        loss = criterion(y_preds, labels)

        # Compute gradients and do step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # record loss
        losses.update(loss.item(), batch_size)
        classes = y_preds.argmax(dim=1)
        acc = torch.mean((classes == labels).float())
        accuracy.update(acc, batch_size)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % CFG.print_freq == 0 or i == (len(train_loader) - 1):
            print(
                "Epoch: [{0}][{1}/{2}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) "
                #'LR: {lr:.6f}  '
                .format(
                    epoch + 1,
                    i,
                    len(train_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(i + 1) / len(train_loader)),
                )
            )
    return losses.avg, accuracy.avg


def valid_fn(valid_loader, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    preds = []
    # switch to evaluation mode
    model.eval()
    start = end = time.time()
    for step, (images, labels) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)

        # compute loss
        with torch.no_grad():
            y_preds = model(images)
        loss = criterion(y_preds, labels)
        losses.update(loss.item(), batch_size)

        # record accuracy
        preds.append(y_preds.softmax(1).to("cpu").numpy())

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader) - 1):
            print(
                "EVAL: [{0}/{1}] "
                "Data {data_time.val:.3f} ({data_time.avg:.3f}) "
                "Elapsed {remain:s} "
                "Loss: {loss.val:.4f}({loss.avg:.4f}) ".format(
                    step,
                    len(valid_loader),
                    batch_time=batch_time,
                    data_time=data_time,
                    loss=losses,
                    remain=timeSince(start, float(step + 1) / len(valid_loader)),
                )
            )
    predictions = np.concatenate(preds)
    return losses.avg, predictions
