from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT


RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "wdbc.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_DATA_PATH = PROCESSED_DIR / "real_train.csv"
TEST_DATA_PATH = PROCESSED_DIR / "real_test.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_csv(RAW_DATA_PATH)

    features = dataset.drop(columns=["malignant"])
    target = dataset["malignant"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    train_data = x_train.copy()
    train_data["malignant"] = y_train

    test_data = x_test.copy()
    test_data["malignant"] = y_test

    train_data.to_csv(TRAIN_DATA_PATH, index=False)
    test_data.to_csv(TEST_DATA_PATH, index=False)

    print("=" * 60)
    print("LEAKAGE-FREE DATA SPLIT CREATED")
    print("=" * 60)
    print(f"Original records : {len(dataset)}")
    print(f"Real train       : {len(train_data)}")
    print(f"Real test        : {len(test_data)}")
    print(f"Train saved at   : {TRAIN_DATA_PATH}")
    print(f"Test saved at    : {TEST_DATA_PATH}")


if __name__ == "__main__":
    main()