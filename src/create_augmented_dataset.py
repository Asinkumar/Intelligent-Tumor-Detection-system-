import pandas as pd
from pathlib import Path

from src.data import PROJECT_ROOT

# Paths
original_path = PROJECT_ROOT / "data" / "raw" / "wdbc.csv"
synthetic_path = PROJECT_ROOT / "data" / "synthetic" / "synthetic_wdbc.csv"

output_dir = PROJECT_ROOT / "data" / "processed"
output_dir.mkdir(parents=True, exist_ok=True)

output_path = output_dir / "augmented_wdbc.csv"

# Load datasets
original = pd.read_csv(original_path)
synthetic = pd.read_csv(synthetic_path)

# Randomly select 1000 synthetic samples
synthetic_subset = synthetic.sample(n=1000, random_state=42)

# Merge datasets
augmented = pd.concat([original, synthetic_subset], ignore_index=True)

# Shuffle dataset
augmented = augmented.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
augmented.to_csv(output_path, index=False)

print("=" * 60)
print("Original Dataset :", original.shape)
print("Synthetic Selected :", synthetic_subset.shape)
print("Final Dataset :", augmented.shape)
print("=" * 60)

print(f"Saved to: {output_path}")