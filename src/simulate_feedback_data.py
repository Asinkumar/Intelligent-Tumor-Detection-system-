from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT


REAL_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_train.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "feedback"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SIMULATED_FEEDBACK_FILE = (
    OUTPUT_DIR
    / "simulated_validated_feedback_dataset.csv"
)

TARGET_COLUMN = "malignant"

N_SIMULATED_SAMPLES = 25

RANDOM_STATE = 42


def main() -> None:

    print("=" * 70)
    print("SIMULATED FEEDBACK DATASET GENERATOR")
    print("=" * 70)

    if not REAL_TRAIN_PATH.exists():

        raise FileNotFoundError(
            f"Training dataset not found: "
            f"{REAL_TRAIN_PATH}"
        )

    real_train = pd.read_csv(
        REAL_TRAIN_PATH
    )

    if TARGET_COLUMN not in real_train.columns:

        raise ValueError(
            f"Target column "
            f"'{TARGET_COLUMN}' not found."
        )

    feature_columns = [
        column
        for column in real_train.columns
        if column != TARGET_COLUMN
    ]

    sample_count = min(
        N_SIMULATED_SAMPLES,
        len(real_train),
    )

    simulated_feedback = (
        real_train
        .sample(
            n=sample_count,
            random_state=RANDOM_STATE,
        )
        .copy()
        .reset_index(drop=True)
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    # Add very small noise only for pipeline testing.
    # This makes the simulated rows distinct from
    # existing training records.
    for column in feature_columns:

        column_std = float(
            real_train[column].std()
        )

        if (
            np.isnan(column_std)
            or column_std == 0
        ):
            continue

        noise = rng.normal(
            loc=0.0,
            scale=column_std * 0.01,
            size=len(simulated_feedback),
        )

        simulated_feedback[column] = (
            simulated_feedback[column]
            + noise
        )

    simulated_feedback.to_csv(
        SIMULATED_FEEDBACK_FILE,
        index=False,
    )

    print(
        f"Simulated validated samples: "
        f"{len(simulated_feedback)}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_columns)}"
    )

    print(
        f"Target column: "
        f"{TARGET_COLUMN}"
    )

    print(
        "\nSimulation file saved to:"
    )

    print(
        SIMULATED_FEEDBACK_FILE
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "Synthetic feedback is for "
        "pipeline testing only."
    )

    print(
        "It must never be mixed with "
        "real doctor feedback."
    )

    print("\n" + "=" * 70)
    print("SIMULATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()