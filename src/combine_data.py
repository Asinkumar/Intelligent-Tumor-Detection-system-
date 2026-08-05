from pathlib import Path

import pandas as pd

from src.data import PROJECT_ROOT

RAW_DATA = PROJECT_ROOT / "data" / "raw" / "wdbc.csv"
SYNTHETIC_DATA = PROJECT_ROOT / "data" / "synthetic" / "synthetic_wdbc.csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "combined_wdbc.csv"


def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(RAW_DATA)
    synthetic = pd.read_csv(SYNTHETIC_DATA)

    combined = pd.concat([original, synthetic], ignore_index=True)

    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    combined.to_csv(OUTPUT_FILE, index=False)

    print("=" * 60)
    print("Combined Dataset Created Successfully")
    print("=" * 60)

    print(f"Original Records  : {len(original)}")
    print(f"Synthetic Records : {len(synthetic)}")
    print(f"Combined Records  : {len(combined)}")

    print()
    print(f"Saved at : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()