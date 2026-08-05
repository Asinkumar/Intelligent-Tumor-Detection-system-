from pathlib import Path

import pandas as pd
from sklearn.datasets import load_breast_cancer


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
RAW_DATA_PATH = RAW_DATA_DIR / "wdbc.csv"

PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_wdbc.csv"


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """
    Load the processed combined dataset (original + synthetic).
    """
    df = pd.read_csv(PROCESSED_DATA_PATH)

    features = df.drop(columns=["malignant"])
    target = df["malignant"]

    return features, target


def export_raw_dataset() -> Path:
    """
    Export the original Wisconsin Breast Cancer dataset.
    """
    dataset = load_breast_cancer(as_frame=True)

    features = dataset.data.copy()
    target = (dataset.target == 0).astype(int)
    target.name = "malignant"

    frame = features.copy()
    frame["malignant"] = target

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(RAW_DATA_PATH, index=False)

    return RAW_DATA_PATH


if __name__ == "__main__":
    path = export_raw_dataset()
    print(f"Dataset exported to: {path}")