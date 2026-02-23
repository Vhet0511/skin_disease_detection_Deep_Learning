Intra-Dataset vs Cross-Dataset Generalization using CNN
📌 Project Overview

This project demonstrates an important concept in Deep Learning:

CNN models perform significantly better when trained and tested on the same dataset distribution compared to when evaluated on a different dataset.

Instead of only building a classifier, this project focuses on understanding model generalization and dataset shift in medical image analysis.

We compare two experiments:

1️⃣ Training and testing on the same dataset (HAM10000 split)
2️⃣ Training on HAM10000 and testing on ISIC 2024 (Cross-dataset evaluation)

📂 Project Structure (Important)

Your folder structure must look exactly like this:
skin_disease_detection/
│
├── data/
│   ├── HAM_binary/
│   │   ├── benign/
│   │   └── malignant/
│   │
│   ├── ISIC_binary/
│   │   ├── benign/
│   │   └── malignant/
│   │
│   ├── HAM10000/
│   └── ISIC_2024_Training_Input/
│
├── src/
│   ├── train_ham_dataset.py
│   ├── train_cross_dataset.py
│   ├── data_analysis/
│   │   ├── inspect_ham.py
│   │   ├── inspect_isic.py
│   │   └── compare_dataset.py
│
├── cross_dataset_model.pth
├── .gitignore
└── README.md

📊 Datasets Used
HAM10000

Total Samples: 10,015

Originally 7 classes

Converted to binary classification:

benign

malignant

Class distribution:

nv: 6705

mel: 1113

bkl: 1099

bcc: 514

akiec: 327

vasc: 142

df: 115

ISIC 2024

Total Samples: 401,059

Binary dataset

0 → benign

1 → malignant

Highly imbalanced:

Benign: 400,666

Malignant: 393

This imbalance plays a major role in cross-dataset results.

Experiment 1: Same Dataset (HAM Train/Test Split)

Run:

python src/train_ham_dataset.py
Training Results

Final Train Accuracy: 94.38%
Test Accuracy: 85%
ROC-AUC Score: 0.907

Confusion Matrix:
[[1406  219]
 [  87  291]]

Experiment 2: Cross-Dataset (Train on HAM → Test on ISIC)

Run:

python src/train_cross_dataset.py

Training Results

Train Accuracy: 91.08%

Testing on ISIC

Accuracy: 99% (misleading due to imbalance)
ROC-AUC Score: 0.68

Confusion Matrix:
[[398661  2005]
 [   377    16]]

How to Run This Project
Step 1: Clone Repository
git clone <your_repo_link>
cd skin_disease_detection
Step 2: Create Virtual Environment

Windows:

python -m venv venv
venv\Scripts\activate
Step 3: Install Dependencies
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow
Step 4: Ensure Data Structure

Make sure images are organized as:

data/HAM_binary/benign/
data/HAM_binary/malignant/
data/ISIC_binary/benign/
data/ISIC_binary/malignant/

Then run the training scripts.
