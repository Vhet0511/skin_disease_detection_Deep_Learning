from pathlib import Path
import pandas as pd
from PIL import Image

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# HAM dataset directory
HAM_DIR = PROJECT_ROOT / "data" / "HAM10000"

# Metadata path
HAM_METADATA_PATH = HAM_DIR / "HAM10000_metadata.csv"

# Image folders
IMAGE_DIR_1 = HAM_DIR / "HAM10000_images_part_1"
IMAGE_DIR_2 = HAM_DIR / "HAM10000_images_part_2"


def find_image_path(image_id):
    img_name = f"{image_id}.jpg"
    path1 = IMAGE_DIR_1 / img_name
    path2 = IMAGE_DIR_2 / img_name

    if path1.exists():
        return path1
    elif path2.exists():
        return path2
    else:
        return None


def check_image_integrity(df, sample_size=50):
    valid = 0
    missing = 0
    corrupted = 0

    sample_df = df.sample(min(sample_size, len(df)), random_state=42)

    for _, row in sample_df.iterrows():
        img_path = find_image_path(row["image_id"])

        if img_path is None:
            missing += 1
            continue

        try:
            Image.open(img_path).verify()
            valid += 1
        except Exception:
            corrupted += 1

    print("\nImage Integrity Check:")
    print(f"Valid images: {valid}")
    print(f"Missing images: {missing}")
    print(f"Corrupted images: {corrupted}")


def main():
    print("Loading HAM10000 metadata...")

    if not HAM_METADATA_PATH.exists():
        raise FileNotFoundError("HAM10000_metadata.csv not found.")

    df = pd.read_csv(HAM_METADATA_PATH)

    print("\nBasic Info:")
    print(df.info())

    print("\nTotal Samples:", len(df))

    if "dx" in df.columns:
        print("\nDiagnosis Distribution:")
        print(df["dx"].value_counts())

    print("\nMissing Values:")
    print(df.isnull().sum())

    check_image_integrity(df)
    
if __name__ == "__main__":
    main()