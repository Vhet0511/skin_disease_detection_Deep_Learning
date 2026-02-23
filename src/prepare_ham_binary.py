import os
import shutil
import pandas as pd

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ham_dir = os.path.join(base_dir, "data", "HAM10000")
metadata_path = os.path.join(ham_dir, "HAM10000_metadata.csv")

images_part1 = os.path.join(ham_dir, "HAM10000_images_part_1")
images_part2 = os.path.join(ham_dir, "HAM10000_images_part_2")

output_dir = os.path.join(base_dir, "data", "HAM_binary")
benign_dir = os.path.join(output_dir, "benign")
malignant_dir = os.path.join(output_dir, "malignant")

os.makedirs(benign_dir, exist_ok=True)
os.makedirs(malignant_dir, exist_ok=True)

df = pd.read_csv(metadata_path)

malignant_labels = ["mel", "bcc", "akiec"]

for _, row in df.iterrows():
    image_name = row["image_id"] + ".jpg"
    label = "malignant" if row["dx"] in malignant_labels else "benign"

    src_path1 = os.path.join(images_part1, image_name)
    src_path2 = os.path.join(images_part2, image_name)

    if os.path.exists(src_path1):
        src_path = src_path1
    elif os.path.exists(src_path2):
        src_path = src_path2
    else:
        continue

    dst_path = os.path.join(output_dir, label, image_name)
    shutil.copy(src_path, dst_path)

print("HAM binary dataset created successfully.")
