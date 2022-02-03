"""
Calculation Of standard deviation and mean (per channel) over all images of the image dataset
"""
import cv2
import pandas as pd

TRAIN_PATH = "../data/cassava-leaf-disease-classification/train_images"
TRAIN_CSV = "../data/cassava-leaf-disease-classification/train.csv"

SIZE = 512


def get_mean_std(train_df):
    sum_mean = 0.0
    sum_std = 0.0
    file_name_list = train_df["image_id"].values
    for file_name in file_name_list:
        print("Processing file ", file_name)
        file_path = f"{TRAIN_PATH}/{file_name}"
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(image, (SIZE, SIZE))
        mean, std = cv2.meanStdDev(resized_image)
        sum_mean += mean
        sum_std += std
    avg_mean = sum_mean / len(file_name_list)
    avg_std = sum_std / len(file_name_list)
    return avg_mean / 255.0, avg_std / 255.0


def main():
    print("Reading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    MEAN, STD = get_mean_std(train_df)
    print("MEAN: ", MEAN, "STD: ", STD)


if __name__ == "__main__":
    main()
