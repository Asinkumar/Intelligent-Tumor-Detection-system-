from __future__ import annotations

from pathlib import Path

import pandas as pd

from src import retraining_trigger


def test_should_retrain_returns_false_when_dataset_missing(
    tmp_path: Path,
) -> None:

    missing_file = (
        tmp_path
        / "missing_feedback.csv"
    )

    result = retraining_trigger.should_retrain(
        missing_file
    )

    assert result is False


def test_should_retrain_returns_false_below_threshold(
    tmp_path: Path,
) -> None:

    feedback_file = (
        tmp_path
        / "feedback.csv"
    )

    dataframe = pd.DataFrame(
        {
            "malignant": [0] * 5
        }
    )

    dataframe.to_csv(
        feedback_file,
        index=False,
    )

    result = retraining_trigger.should_retrain(
        feedback_file
    )

    assert result is False


def test_should_retrain_returns_true_at_threshold(
    tmp_path: Path,
) -> None:

    feedback_file = (
        tmp_path
        / "feedback.csv"
    )

    dataframe = pd.DataFrame(
        {
            "malignant": (
                [0]
                * retraining_trigger.MIN_FEEDBACK_SAMPLES
            )
        }
    )

    dataframe.to_csv(
        feedback_file,
        index=False,
    )

    result = retraining_trigger.should_retrain(
        feedback_file
    )

    assert result is True