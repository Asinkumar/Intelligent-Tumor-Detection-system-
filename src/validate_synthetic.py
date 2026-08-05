from pathlib import Path
import pandas as pd

from src.data import PROJECT_ROOT

original = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "wdbc.csv")
synthetic = pd.read_csv(PROJECT_ROOT / "data" / "synthetic" / "synthetic_wdbc.csv")

print("=" * 60)
print("Original Dataset")
print(original.describe())

print("=" * 60)
print("Synthetic Dataset")
print(synthetic.describe())

print("=" * 60)
print("Original Shape :", original.shape)
print("Synthetic Shape :", synthetic.shape)

print("=" * 60)
print("Missing Values")
print(synthetic.isnull().sum().sum())

print("=" * 60)
print("Duplicate Rows")
print(synthetic.duplicated().sum())