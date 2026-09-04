from __future__ import annotations

import argparse

from src.build_feedback_dataset import main as build_feedback_dataset
from src.monitoring import run_monitoring
from src.retrain_model import retrain_model
from src.retraining_trigger import (
    REAL_FEEDBACK_DATASET,
    SIMULATED_FEEDBACK_DATASET,
    should_retrain,
)


def run_pipeline(
    simulate: bool = False,
) -> None:

    print("=" * 70)
    print("CONTINUOUS LEARNING MLOPS PIPELINE")
    print("=" * 70)

    if simulate:

        feedback_dataset = (
            SIMULATED_FEEDBACK_DATASET
        )

        print("\nTEST MODE ENABLED")
        print(
            "Using simulated validated feedback."
        )

    else:

        feedback_dataset = (
            REAL_FEEDBACK_DATASET
        )

        print("\nPRODUCTION MODE")
        print(
            "Using real doctor-validated feedback."
        )

        print(
            "\nSTEP 1: BUILD VALIDATED FEEDBACK DATASET"
        )

        print("-" * 70)

        build_feedback_dataset()

    # ---------------------------------------------------
    # STEP 2: Monitoring
    # ---------------------------------------------------

    print(
        "\nSTEP 2: DATA DRIFT MONITORING"
    )

    print("-" * 70)

    run_monitoring(
        current_dataset_path=feedback_dataset,
        test_mode=simulate,
    )

    # ---------------------------------------------------
    # STEP 3: Retraining gate
    # ---------------------------------------------------

    print(
        "\nSTEP 3: RETRAINING TRIGGER"
    )

    print("-" * 70)

    ready = should_retrain(
        feedback_dataset
    )

    if not ready:

        print(
            "\nPipeline stopped safely."
        )

        print(
            "Not enough validated feedback "
            "samples for retraining."
        )

        print("\n" + "=" * 70)
        print(
            "CONTINUOUS LEARNING PIPELINE COMPLETED"
        )
        print("=" * 70)

        return

    # ---------------------------------------------------
    # STEP 4: Retraining + candidate evaluation
    # ---------------------------------------------------

    print(
        "\nSTEP 4: CANDIDATE RETRAINING"
    )

    print("-" * 70)

    retrain_model(
        feedback_dataset=feedback_dataset,
        test_mode=simulate,
    )

    print("\n" + "=" * 70)
    print(
        "CONTINUOUS LEARNING PIPELINE COMPLETED"
    )
    print("=" * 70)


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Continuous learning pipeline for "
            "Tumor Decision Support System"
        )
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Run the complete continuous-learning "
            "pipeline with simulated feedback."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    run_pipeline(
        simulate=args.simulate
    )


if __name__ == "__main__":
    main()