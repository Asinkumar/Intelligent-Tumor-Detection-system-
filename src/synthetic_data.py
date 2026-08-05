from pathlib import Path

import pandas as pd
from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer

from src.data import PROJECT_ROOT


RAW_DATA = PROJECT_ROOT / "data" / "processed" / "real_train.csv"
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"
MODEL_DIR = PROJECT_ROOT / "models"

N_SYNTHETIC = 1000


def main():

    SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Loading original dataset...")
    print("=" * 60)

    df = pd.read_csv(RAW_DATA)

    print(df.head())
    print()

    print(f"Original Shape : {df.shape}")
    print()

    print("Creating metadata...")

    metadata = Metadata.detect_from_dataframe(df)

    print("Metadata created successfully.")
    print()

    print("=" * 60)
    print("Training CTGAN Synthesizer...")
    print("=" * 60)

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=300,
        verbose=True
    )

    synthesizer.fit(df)

    print()
    print("Training Completed.")
    print()

    print("=" * 60)
    print(f"Generating {N_SYNTHETIC} Synthetic Records...")
    print("=" * 60)

    synthetic_df = synthesizer.sample(num_rows=N_SYNTHETIC)

    output_file = SYNTHETIC_DIR / "synthetic_wdbc.csv"

    synthetic_df.to_csv(output_file, index=False)

    model_file = MODEL_DIR / "ctgan_synthesizer.pkl"

    synthesizer.save(filepath=model_file)

    print()
    print("=" * 60)
    print("Generation Completed Successfully")
    print("=" * 60)

    print(f"Original Dataset Shape : {df.shape}")
    print(f"Synthetic Dataset Shape : {synthetic_df.shape}")

    print()

    print("Synthetic Dataset Preview")

    print(synthetic_df.head())

    print()

    print(f"Synthetic dataset saved to : {output_file}")

    print(f"CTGAN model saved to : {model_file}")


if __name__ == "__main__":
    main()