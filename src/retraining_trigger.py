from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import PROJECT_ROOT


REAL_FEEDBACK_DATASET = (
    PROJECT_ROOT
    / "data"
    / "feedback"
    / "validated_feedback_dataset.csv"
)

SIMULATED_FEEDBACK_DATASET = (
    PROJECT_ROOT
    / "data"
    / "feedback"
    / "simulated_validated_feedback_dataset.csv"
)

MIN_FEEDBACK_SAMPLES = 20


def should_retrain(
    dataset_path: Path | None = None,
) -> bool:

    if dataset_path is None:
        dataset_path = REAL_FEEDBACK_DATASET

    if not dataset_path.exists():

        print(
            f"Feedback dataset not found: "
            f"{dataset_path}"
        )

        return False

    feedback_df = pd.read_csv(
        dataset_path
    )

    sample_count = len(
        feedback_df
    )

    print(
        f"Feedback dataset: "
        f"{dataset_path.name}"
    )

    print(
        f"Validated feedback samples: "
        f"{sample_count}"
    )

    print(
        f"Minimum samples required: "
        f"{MIN_FEEDBACK_SAMPLES}"
    )

    if sample_count >= MIN_FEEDBACK_SAMPLES:

        print(
            "Retraining trigger: READY"
        )

        return True

    remaining = (
        MIN_FEEDBACK_SAMPLES
        - sample_count
    )

    print(
        "Retraining trigger: NOT READY"
    )

    print(
        f"Additional validated samples needed: "
        f"{remaining}"
    )

    return False


def main() -> None:

    print("=" * 70)
    print("RETRAINING TRIGGER CHECK")
    print("=" * 70)

    should_retrain(
        REAL_FEEDBACK_DATASET
    )


if __name__ == "__main__":
    main()