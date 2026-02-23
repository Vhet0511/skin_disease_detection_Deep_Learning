import os
import pandas as pd
from PIL import Image

ISIC_METADATA_PATH = "data/ISIC_2024_Training_Input/metadata.csv"
ISIC_IMAGE_DIR = "data/ISIC_2024_Training_Input"


def check_image_integrity(image_dir, sample_size=50):
    valid = 0
    corrupted = 0
    files = os.listdir(image_dir)[:sample_size]

    for file in files:
        try:
            img_path = os.path.join(image_dir, file)
            Image.open(img_path).verify()
            valid += 1
        except Exception:
            corrupted += 1

    print(f"Image Integrity Check (sample {sample_size}):")
    print(f"Valid images: {valid}")
    print(f"Corrupted images: {corrupted}")


def main():
    print("Loading ISIC metadata...")
    df = pd.read_csv(ISIC_METADATA_PATH)

    print("\nBasic Info:")
    print(df.info())

    print("\nTotal Samples:", len(df))

    if "diagnosis" in df.columns:
        print("\nDiagnosis Distribution:")
        print(df["diagnosis"].value_counts())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nChecking image integrity...")
    check_image_integrity(ISIC_IMAGE_DIR)


if __name__ == "__main__":
    main()
