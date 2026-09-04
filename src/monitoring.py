from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.data import PROJECT_ROOT
from src.retraining_trigger import (
    REAL_FEEDBACK_DATASET,
    SIMULATED_FEEDBACK_DATASET,
)


# =======================================================
# CONFIGURATION
# =======================================================

REFERENCE_DATASET = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_train.csv"
)

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "monitoring"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

REPORT_FILE = (
    REPORT_DIR
    / "data_drift_report.csv"
)

SUMMARY_FILE = (
    REPORT_DIR
    / "data_drift_summary.txt"
)

TARGET_COLUMN = "malignant"

KS_PVALUE_THRESHOLD = 0.05

MEAN_SHIFT_THRESHOLD = 0.50

DRIFTED_FEATURE_RATIO_THRESHOLD = 0.20


# =======================================================
# LOAD DATA
# =======================================================

def load_datasets(
    current_dataset_path: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
]:

    if not REFERENCE_DATASET.exists():

        raise FileNotFoundError(
            f"Reference training dataset not found: "
            f"{REFERENCE_DATASET}"
        )

    if not current_dataset_path.exists():

        raise FileNotFoundError(
            f"Current monitoring dataset not found: "
            f"{current_dataset_path}"
        )

    reference_df = pd.read_csv(
        REFERENCE_DATASET
    )

    current_df = pd.read_csv(
        current_dataset_path
    )

    print(
        f"Reference samples : "
        f"{len(reference_df)}"
    )

    print(
        f"Current samples   : "
        f"{len(current_df)}"
    )

    return (
        reference_df,
        current_df,
    )


# =======================================================
# VALIDATE FEATURES
# =======================================================

def get_feature_columns(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> list[str]:

    feature_columns = [
        column
        for column in reference_df.columns
        if column != TARGET_COLUMN
    ]

    missing_columns = [
        column
        for column in feature_columns
        if column not in current_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Current dataset is missing model features: "
            f"{missing_columns}"
        )

    print(
        f"Features monitored : "
        f"{len(feature_columns)}"
    )

    return feature_columns


# =======================================================
# FEATURE DRIFT CHECK
# =======================================================

def analyze_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:

    results = []

    for feature in feature_columns:

        reference_values = (
            reference_df[
                feature
            ]
            .dropna()
            .astype(float)
        )

        current_values = (
            current_df[
                feature
            ]
            .dropna()
            .astype(float)
        )

        if (
            reference_values.empty
            or current_values.empty
        ):

            continue

        reference_mean = float(
            reference_values.mean()
        )

        current_mean = float(
            current_values.mean()
        )

        reference_std = float(
            reference_values.std()
        )

        mean_difference = (
            current_mean
            - reference_mean
        )

        if (
            np.isnan(reference_std)
            or reference_std == 0
        ):

            normalized_mean_shift = 0.0

        else:

            normalized_mean_shift = (
                abs(mean_difference)
                / reference_std
            )

        ks_result = ks_2samp(
            reference_values,
            current_values,
        )

        ks_statistic = float(
            ks_result.statistic
        )

        ks_pvalue = float(
            ks_result.pvalue
        )

        ks_drift = (
            ks_pvalue
            < KS_PVALUE_THRESHOLD
        )

        mean_shift_drift = (
            normalized_mean_shift
            >= MEAN_SHIFT_THRESHOLD
        )

        drift_detected = (
            ks_drift
            or mean_shift_drift
        )

        results.append(
            {
                "Feature": feature,
                "Reference_Mean": reference_mean,
                "Current_Mean": current_mean,
                "Reference_Std": reference_std,
                "Normalized_Mean_Shift": (
                    normalized_mean_shift
                ),
                "KS_Statistic": ks_statistic,
                "KS_PValue": ks_pvalue,
                "KS_Drift": ks_drift,
                "Mean_Shift_Drift": (
                    mean_shift_drift
                ),
                "Drift_Detected": (
                    drift_detected
                ),
            }
        )

    return pd.DataFrame(
        results
    )


# =======================================================
# OVERALL DRIFT STATUS
# =======================================================

def calculate_overall_status(
    drift_df: pd.DataFrame,
) -> tuple[
    int,
    float,
    bool,
]:

    if drift_df.empty:

        return (
            0,
            0.0,
            False,
        )

    drifted_features = int(
        drift_df[
            "Drift_Detected"
        ].sum()
    )

    total_features = len(
        drift_df
    )

    drift_ratio = (
        drifted_features
        / total_features
    )

    overall_drift = (
        drift_ratio
        >= DRIFTED_FEATURE_RATIO_THRESHOLD
    )

    return (
        drifted_features,
        drift_ratio,
        overall_drift,
    )


# =======================================================
# SAVE REPORT
# =======================================================

def save_monitoring_report(
    drift_df: pd.DataFrame,
    current_dataset_path: Path,
) -> None:

    (
        drifted_features,
        drift_ratio,
        overall_drift,
    ) = calculate_overall_status(
        drift_df
    )

    drift_df.to_csv(
        REPORT_FILE,
        index=False,
    )

    status = (
        "DRIFT DETECTED"
        if overall_drift
        else "NO SIGNIFICANT DRIFT"
    )

    summary = (
        "=" * 70
        + "\nDATA DRIFT MONITORING SUMMARY\n"
        + "=" * 70
        + "\n"
        + f"Reference Dataset : {REFERENCE_DATASET}\n"
        + f"Current Dataset   : {current_dataset_path}\n"
        + f"Features Checked  : {len(drift_df)}\n"
        + f"Drifted Features  : {drifted_features}\n"
        + f"Drift Ratio       : {drift_ratio:.2%}\n"
        + f"Overall Status    : {status}\n"
        + "=" * 70
        + "\n"
    )

    SUMMARY_FILE.write_text(
        summary,
        encoding="utf-8",
    )

    print(
        "\n" + summary
    )

    print(
        "Detailed monitoring report saved to:"
    )

    print(
        REPORT_FILE
    )

    print(
        "\nSummary saved to:"
    )

    print(
        SUMMARY_FILE
    )


# =======================================================
# MONITORING WORKFLOW
# =======================================================

def run_monitoring(
    current_dataset_path: Path,
    test_mode: bool = False,
) -> None:

    print("=" * 70)
    print("MODEL DATA DRIFT MONITORING")
    print("=" * 70)

    if test_mode:

        print(
            "\nTEST MODE ENABLED"
        )

        print(
            "Monitoring simulated feedback data."
        )

    else:

        print(
            "\nPRODUCTION MODE"
        )

        print(
            "Monitoring real doctor-validated feedback data."
        )

    print(
        f"\nCurrent monitoring dataset: "
        f"{current_dataset_path}"
    )

    (
        reference_df,
        current_df,
    ) = load_datasets(
        current_dataset_path
    )

    feature_columns = (
        get_feature_columns(
            reference_df=reference_df,
            current_df=current_df,
        )
    )

    drift_df = (
        analyze_feature_drift(
            reference_df=reference_df,
            current_df=current_df,
            feature_columns=feature_columns,
        )
    )

    save_monitoring_report(
        drift_df=drift_df,
        current_dataset_path=current_dataset_path,
    )

    print("\nTop potentially drifting features")
    print("-" * 70)

    top_drift = (
        drift_df
        .sort_values(
            [
                "Drift_Detected",
                "Normalized_Mean_Shift",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .head(10)
    )

    if top_drift.empty:

        print(
            "No feature statistics available."
        )

    else:

        print(
            top_drift[
                [
                    "Feature",
                    "Normalized_Mean_Shift",
                    "KS_PValue",
                    "Drift_Detected",
                ]
            ]
            .to_string(
                index=False
            )
        )

    print("\n" + "=" * 70)
    print("MONITORING COMPLETED")
    print("=" * 70)


# =======================================================
# CLI
# =======================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Data drift monitoring for "
            "Tumor Decision Support System"
        )
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Monitor simulated feedback "
            "instead of real doctor feedback."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    if args.simulate:

        run_monitoring(
            current_dataset_path=(
                SIMULATED_FEEDBACK_DATASET
            ),
            test_mode=True,
        )

    else:

        run_monitoring(
            current_dataset_path=(
                REAL_FEEDBACK_DATASET
            ),
            test_mode=False,
        )


if __name__ == "__main__":
    main()