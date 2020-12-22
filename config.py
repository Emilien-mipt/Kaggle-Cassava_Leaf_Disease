import pandas as pd


class CFG:
    # Data path
    TRAIN_PATH = "./data/cassava-leaf-disease-classification/train_images"
    TEST_PATH = "./data/cassava-leaf-disease-classification/test_images"
    OUTPUT_DIR = "./logs"

    # Main config
    GPU_ID = 1
    seed = 42
    target_size = 5
    target_col = "label"
    trn_fold = [0, 1, 2, 3, 4]
    train = True
    inference = False

    # Train configs
    debug = False
    apex = False
    epochs = 50
    model_name = "tf_efficientnet_b3_ns"
    batch_size = 32
    size = 256
    num_workers = 8
    print_freq = 100
    # Optimizer config
    lr = 1e-3
    min_lr = 1e-6
    weight_decay = 1e-6
    gradient_accumulation_steps = 1
    max_grad_norm = 1000


if CFG.debug:
    CFG.epochs = 1
    train_pd = pd.read_csv("../input/cassava-leaf-disease-classification/train.csv")
    train = train_pd.sample(n=1000, random_state=CFG.seed).reset_index(drop=True)
