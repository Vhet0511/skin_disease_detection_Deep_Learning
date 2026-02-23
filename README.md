# Intra-Dataset vs Cross-Dataset Generalization using CNN

##  Project Overview

This project demonstrates an important concept in Deep Learning:

> CNN models perform significantly better when trained and tested on the same dataset distribution compared to when evaluated on a different dataset.

Instead of only building a classifier, this project focuses on understanding:

- Model Generalization  
- Dataset Shift  
- Domain Distribution Differences in Medical Image Analysis  

We compare two experiments:

1. Training and testing on the same dataset (HAM10000 split)  
2. Training on HAM10000 and testing on ISIC 2024 (Cross-dataset evaluation)  

---

## 📂 Project Structure (Important)

Your folder structure must look exactly like this:

```
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
```

---

##  Datasets Used

### HAM10000

- Total Samples: 10,015  
- Originally 7 classes  
- Converted to Binary Classification:
  - benign  
  - malignant  

### Class Distribution:

| Class | Samples |
|-------|---------|
| nv    | 6705    |
| mel   | 1113    |
| bkl   | 1099    |
| bcc   | 514     |
| akiec | 327     |
| vasc  | 142     |
| df    | 115     |

---

### ISIC 2024

- Total Samples: 401,059  
- Binary dataset:
  - 0 → benign  
  - 1 → malignant  

### Class Distribution:

| Class      | Samples |
|------------|----------|
| Benign     | 400,666  |
| Malignant  | 393      |

⚠️ This extreme imbalance plays a major role in cross-dataset results.

---

#  Experiment 1: Same Dataset (HAM Train/Test Split)

### Run:

```
python src/train_ham_dataset.py
```

### Training Results

- Final Train Accuracy: 94.38%  
- Test Accuracy: 85%  
- ROC-AUC Score: 0.907  

### Confusion Matrix:

```
[[1406  219]
 [  87  291]]
```

---

#  Experiment 2: Cross-Dataset (Train on HAM → Test on ISIC)

### Run:

```
python src/train_cross_dataset.py
```

### Training Results

- Train Accuracy: 91.08%

### Testing on ISIC

- Accuracy: 99% (misleading due to imbalance)  
- ROC-AUC Score: 0.68  

### Confusion Matrix:

```
[[398661  2005]
 [   377    16]]
```

⚠️ Notice how accuracy appears extremely high but ROC-AUC drops significantly.  
This highlights the dataset shift problem.

---

# How to Run This Project

## Step 1: Clone Repository

```
git clone <your_repo_link>
cd skin_disease_detection
```

## Step 2: Create Virtual Environment

### Windows:

```
python -m venv venv
venv\Scripts\activate
```

## Step 3: Install Dependencies

```
pip install torch torchvision pandas numpy matplotlib scikit-learn pillow
```

## Step 4: Ensure Data Structure

Make sure images are organized as:

```
data/HAM_binary/benign/
data/HAM_binary/malignant/
data/ISIC_binary/benign/
data/ISIC_binary/malignant/
```

Then run the training scripts.

---

# Key Takeaways

- Same-dataset evaluation gives strong performance.
- Cross-dataset evaluation reveals real-world generalization challenges.
- Accuracy alone is not reliable in highly imbalanced medical datasets.
- ROC-AUC is a better metric for evaluating generalization.

---

# 10 Reasons Lab-Tested CNN Models Fail in Real-World Settings

1. Distribution Shift
Training data and real-world data follow different statistical distributions. When 
𝑃
𝑡
𝑟
𝑎
𝑖
𝑛
(
𝑋
,
𝑌
)
≠
𝑃
𝑟
𝑒
𝑎
𝑙
(
𝑋
,
𝑌
)
P
train
	​

(X,Y)

=P
real
	​

(X,Y), the learned decision boundary no longer aligns with reality.

2. Severe Class Imbalance
Real-world datasets often have extreme imbalance (like ISIC 2024). Accuracy becomes misleading because predicting the majority class yields high scores while missing critical minority cases.

3. Overfitting to Dataset-Specific Artifacts
CNNs may learn background textures, lighting conditions, or camera signatures instead of true pathological features.

4. Different Image Acquisition Pipelines
Variation in devices, resolution, zoom level, dermatoscopes, or hospitals introduces systematic visual differences the model was never trained on.

5. Label Noise in Real Data
Lab datasets are often curated and verified. Real-world labels may contain annotation errors, uncertain diagnoses, or inconsistent labeling standards.

6. Hidden Confounders
The model may unintentionally learn shortcuts (e.g., image borders, watermark patterns, demographic correlations) that don’t generalize.

7. Small Effective Sample Diversity
Even large datasets may lack diversity in skin tones, geographic populations, disease stages, or imaging conditions.

8. Domain Over-Specialization
Models trained on one dataset specialize to that domain’s visual characteristics and fail when the domain changes (cross-domain generalization issue).

9. Metric Misinterpretation
Accuracy hides failure in minority classes. Metrics like ROC-AUC, precision-recall curves, and F1-score better reflect real diagnostic performance.

10. Real-World Complexity & Noise
Clinical settings introduce blur, occlusion, shadows, partial lesions, and non-standard framing that clean lab datasets rarely include.
