# Cassava Leaf Disease Classification

## Problem overview
As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated.

Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As an added challenge, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth.

In this competition, we introduce a dataset of 21,367 labeled images collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.

The task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

![2216849948](https://user-images.githubusercontent.com/44554040/152436810-b938884a-e60c-40e6-9920-5ebd48975521.jpg)
*Example of the image in the test set*

## Code structure
This repository introduces the baseline that was developed to solve the Kaggle competition
"Cassava Leaf Disease Classification". Though the code was written for solving the competition, due to the universality
of it's structure it can be also easily generelized to any classification problem in computer vision.
The structure goes as follows:

#### Config file
All the settings (data paths and train parameters) are set in `CFG` class, which is located in `config.py` file.

#### Model
The baseline is pretty straight-forward: the computer vision model is being
set via the [PyTorch Image Models](https://pypi.org/project/timm/) library and the last classifier layer is changed according to the
number of output classes (In our case it is 5). The model is set in `model.py` file, while the type of the
architecture is set in config file.

#### Training and inference
After the parameters for training are set in config file,
the training can be run with the following command:

```python run_trainer.py --logdir_name=...```,

where logdir_name is the name of the directory, where all the logs (log file with selected train parameters,
tensorboard plots for metrics and model weigths) are saved.

If you would like to save the images after augmentations, add `-save_batch_fig` to the train command.

If you would like to run experiments to find optimal learning rate, add `--find_lr` to the train command.

Inference and prediction code on the test set is located in `inference.py` and `predict.py` code correspondingly.

#### Augmentations
Augmentations are set via the [Albumentations](https://albumentations.ai/) library in the `augmentations.py` file.
