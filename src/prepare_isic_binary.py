import os
import shutil
import pandas as pd

base_dir = r"C:\Users\hetvy\OneDrive\Documents\skin_disease_detection\data"

isic_dir = os.path.join(base_dir, "ISIC_2024_Training_Input")
metadata_path = os.path.join(isic_dir, "ISIC_2024_Training_GroundTruth.csv")

output_dir = os.path.join(base_dir, "ISIC_binary")
benign_dir = os.path.join(output_dir, "benign")
malignant_dir = os.path.join(output_dir, "malignant")

os.makedirs(benign_dir, exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)

df = pd.read_csv(metadata_path)

print("Total rows in CSV:", len(df))

copied = 0
missing = 0

for _, row in df.iterrows():
    image_id = row["isic_id"]
    label = "malignant" if row["malignant"] == 1 else "benign"
    image_name = image_id + ".jpg"

    src_path = os.path.join(isic_dir, image_name)
    dst_path = os.path.join(output_dir, label, image_name)

    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
        copied += 1
    else:
        missing += 1

print("Copied images:", copied)
print("Missing images:", missing)
print("ISIC binary dataset created successfully.")
