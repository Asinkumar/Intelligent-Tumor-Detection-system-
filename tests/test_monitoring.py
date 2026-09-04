from __future__ import annotations

import pandas as pd

from src.monitoring import (
    analyze_feature_drift,
    calculate_overall_status,
)


def test_no_drift_for_identical_distributions() -> None:

    reference_df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [10, 11, 12, 13, 14],
            "malignant": [0, 0, 1, 1, 0],
        }
    )

    current_df = reference_df.copy()

    feature_columns = [
        "feature_a",
        "feature_b",
    ]

    drift_df = analyze_feature_drift(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=feature_columns,
    )

    drifted_features, drift_ratio, overall_drift = (
        calculate_overall_status(
            drift_df
        )
    )

    assert drifted_features == 0
    assert drift_ratio == 0.0
    assert overall_drift is False


def test_detects_strong_feature_drift() -> None:

    reference_df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [10, 11, 12, 13, 14],
            "malignant": [0, 0, 1, 1, 0],
        }
    )

    current_df = pd.DataFrame(
        {
            "feature_a": [100, 110, 120, 130, 140],
            "feature_b": [200, 210, 220, 230, 240],
            "malignant": [0, 1, 1, 0, 1],
        }
    )

    feature_columns = [
        "feature_a",
        "feature_b",
    ]

    drift_df = analyze_feature_drift(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=feature_columns,
    )

    drifted_features, drift_ratio, overall_drift = (
        calculate_overall_status(
            drift_df
        )
    )

    assert drifted_features == 2
    assert drift_ratio == 1.0
    assert overall_drift is True


def test_partial_drift_below_overall_threshold() -> None:

    reference_df = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [10, 11, 12, 13, 14],
            "feature_c": [20, 21, 22, 23, 24],
            "feature_d": [30, 31, 32, 33, 34],
            "feature_e": [40, 41, 42, 43, 44],
            "malignant": [0, 0, 1, 1, 0],
        }
    )

    current_df = reference_df.copy()

    current_df["feature_a"] = [
        100,
        110,
        120,
        130,
        140,
    ]

    feature_columns = [
        "feature_a",
        "feature_b",
        "feature_c",
        "feature_d",
        "feature_e",
    ]

    drift_df = analyze_feature_drift(
        reference_df=reference_df,
        current_df=current_df,
        feature_columns=feature_columns,
    )

    drifted_features, drift_ratio, overall_drift = (
        calculate_overall_status(
            drift_df
        )
    )

    assert drifted_features == 1
    assert drift_ratio == 0.2
    assert overall_drift is True