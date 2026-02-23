from pathlib import Path
import pandas as pd

# Resolve project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ================= PATHS =================
ISIC_DIR = PROJECT_ROOT / "data" / "ISIC_2024_Training_Input"
HAM_DIR = PROJECT_ROOT / "data" / "HAM10000"

ISIC_GROUND_TRUTH = ISIC_DIR / "ISIC_2024_Training_GroundTruth.csv"
HAM_METADATA = HAM_DIR / "HAM10000_metadata.csv"
# =========================================


def load_isic_labels():
    if not ISIC_GROUND_TRUTH.exists():
        raise FileNotFoundError("ISIC GroundTruth file not found.")

    df = pd.read_csv(ISIC_GROUND_TRUTH)

    if "malignant" not in df.columns:
        raise ValueError("ISIC GroundTruth does not contain 'malignant' column.")

    return df["malignant"]


def load_ham_labels():
    if not HAM_METADATA.exists():
        raise FileNotFoundError("HAM metadata file not found.")

    df = pd.read_csv(HAM_METADATA)

    if "dx" not in df.columns:
        raise ValueError("HAM metadata does not contain 'dx' column.")

    return df["dx"]


def main():
    print("Loading ISIC 2024 Ground Truth...")
    isic_labels = load_isic_labels()

    print("Loading HAM10000 Metadata...")
    ham_labels = load_ham_labels()

    print("\n========== ISIC 2024 ==========")
    print("Total Samples:", len(isic_labels))
    print("Class Distribution:")
    print(isic_labels.value_counts())

    print("\n========== HAM10000 ==========")
    print("Total Samples:", len(ham_labels))
    print("Class Distribution:")
    print(ham_labels.value_counts())

    print("\n========== SUMMARY ==========")
    print(f"ISIC unique classes: {isic_labels.nunique()} (Binary)")
    print(f"HAM unique classes: {ham_labels.nunique()} (Multi-class)")

    print("\nNOTE:")
    print("ISIC is binary (0 = benign, 1 = malignant).")
    print("HAM must be mapped to binary to compare properly.")


if __name__ == "__main__":
    main()
