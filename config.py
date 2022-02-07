import pandas as pd


class CFG:
    # Data path
    TRAIN_PATH = "./data/cassava-leaf-disease-classification/train_images"
    TRAIN_CSV = "./data/cassava-leaf-disease-classification/train.csv"
    TEST_PATH = "./data/cassava-leaf-disease-classification/test_images"
    OUTPUT_DIR = "./logs"

    # Main config
    GPU_ID = 0
    seed = 42
    target_size = 5
    target_col = "label"

    # Train configs
    MIXED_PREC = True  # Flag for mixed precision training
    debug = False
    test_size = 0.2
    epochs = 50
    early_stopping = 10
    model_name = "efficientnet_b3a"
    pretrain = True
    batch_size = 16
    size = 512
    MEAN = [0.485, 0.456, 0.406]  # ImageNet values
    STD = [0.229, 0.224, 0.225]  # ImageNet values
    num_workers = 8
    print_freq = 100

    # Optimizer config
    lr = 1e-1
    momentum = 0.9
    min_lr = 1e-3
    weight_decay = 1e-6

    # Criterion config
    # Label smoothing
    criterion = "LabelSmoothing"
    smooth_alpha = 0.4
