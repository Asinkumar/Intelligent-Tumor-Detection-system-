from __future__ import annotations

import pandas as pd

from src.data import PROJECT_ROOT


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PREDICTION_DIR = (
    PROJECT_ROOT
    / "reports"
    / "predictions"
)

FEEDBACK_FILE = (
    PROJECT_ROOT
    / "reports"
    / "feedback"
    / "doctor_feedback.csv"
)

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

OUTPUT_FILE = (
    OUTPUT_DIR
    / "validated_feedback_dataset.csv"
)

TARGET_COLUMN = "malignant"


# -------------------------------------------------------
# Load exact model feature names
# -------------------------------------------------------

def load_feature_names() -> list[str]:

    if not REAL_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {REAL_TRAIN_PATH}"
        )

    columns = pd.read_csv(
        REAL_TRAIN_PATH,
        nrows=0,
    ).columns.tolist()

    feature_names = [
        column
        for column in columns
        if column != TARGET_COLUMN
    ]

    print(
        f"Expected model features: "
        f"{len(feature_names)}"
    )

    return feature_names


# -------------------------------------------------------
# Load doctor feedback
# -------------------------------------------------------

def load_feedback() -> pd.DataFrame:

    if not FEEDBACK_FILE.exists():
        raise FileNotFoundError(
            f"Feedback file not found: {FEEDBACK_FILE}"
        )

    feedback_df = pd.read_csv(
        FEEDBACK_FILE
    )

    required_columns = [
        "Case_ID",
        "Doctor_Name",
        "Final_Clinical_Decision",
        "Model_Clinical_Agreement",
        "Clinical_Comments",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in feedback_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feedback file is missing required columns: "
            f"{missing_columns}"
        )

    print(
        f"Feedback records found: "
        f"{len(feedback_df)}"
    )

    return feedback_df


# -------------------------------------------------------
# Load prediction records
# -------------------------------------------------------

def load_prediction_records() -> pd.DataFrame:

    prediction_files = list(
        PREDICTION_DIR.glob(
            "*_prediction.csv"
        )
    )

    if not prediction_files:
        raise FileNotFoundError(
            f"No prediction files found in: {PREDICTION_DIR}"
        )

    records = []

    for prediction_file in prediction_files:

        try:

            prediction_df = pd.read_csv(
                prediction_file
            )

        except Exception as error:

            print(
                f"Skipping unreadable file: "
                f"{prediction_file.name}"
            )

            print(
                f"Reason: {error}"
            )

            continue

        if "Case_ID" not in prediction_df.columns:

            print(
                f"Skipping prediction file without Case_ID: "
                f"{prediction_file.name}"
            )

            continue

        records.append(
            prediction_df
        )

    if not records:
        raise ValueError(
            "No valid prediction records found."
        )

    combined_predictions = pd.concat(
        records,
        ignore_index=True,
        sort=False,
    )

    print(
        f"Prediction records found: "
        f"{len(combined_predictions)}"
    )

    return combined_predictions


# -------------------------------------------------------
# Join predictions and doctor feedback
# -------------------------------------------------------

def build_validated_feedback_dataset(
    predictions_df: pd.DataFrame,
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:

    feedback_columns = [
        "Case_ID",
        "Doctor_Name",
        "Final_Clinical_Decision",
        "Model_Clinical_Agreement",
        "Clinical_Comments",
    ]

    feedback_subset = (
        feedback_df[
            feedback_columns
        ]
        .copy()
    )

    merged_df = predictions_df.merge(
        feedback_subset,
        on="Case_ID",
        how="inner",
    )

    print(
        f"Prediction + feedback matches: "
        f"{len(merged_df)}"
    )

    return merged_df


# -------------------------------------------------------
# Keep only rows containing all 30 features
# -------------------------------------------------------

def filter_complete_feature_rows(
    merged_df: pd.DataFrame,
    feature_names: list[str],
) -> pd.DataFrame:

    missing_feature_columns = [
        column
        for column in feature_names
        if column not in merged_df.columns
    ]

    if missing_feature_columns:
        raise ValueError(
            "Prediction records are missing model features: "
            f"{missing_feature_columns}"
        )

    complete_rows = merged_df.dropna(
        subset=feature_names
    ).copy()

    print(
        f"Complete 30-feature feedback records: "
        f"{len(complete_rows)}"
    )

    return complete_rows


# -------------------------------------------------------
# Convert clinical decision to training label
# -------------------------------------------------------

def add_training_label(
    feedback_df: pd.DataFrame,
) -> pd.DataFrame:

    labeled_df = feedback_df[
        feedback_df[
            "Final_Clinical_Decision"
        ].isin(
            [
                "Benign",
                "Malignant",
            ]
        )
    ].copy()

    label_map = {
        "Benign": 0,
        "Malignant": 1,
    }

    labeled_df[
        TARGET_COLUMN
    ] = (
        labeled_df[
            "Final_Clinical_Decision"
        ]
        .map(label_map)
        .astype(int)
    )

    print(
        f"Usable doctor-labeled records: "
        f"{len(labeled_df)}"
    )

    return labeled_df


# -------------------------------------------------------
# Remove duplicate Case IDs
# -------------------------------------------------------

def remove_duplicate_cases(
    labeled_df: pd.DataFrame,
) -> pd.DataFrame:

    before = len(
        labeled_df
    )

    cleaned_df = (
        labeled_df
        .drop_duplicates(
            subset=["Case_ID"],
            keep="last",
        )
        .copy()
    )

    removed = (
        before
        - len(cleaned_df)
    )

    print(
        f"Duplicate cases removed: "
        f"{removed}"
    )

    return cleaned_df


# -------------------------------------------------------
# Save retraining-ready dataset
# -------------------------------------------------------

def save_validated_dataset(
    labeled_df: pd.DataFrame,
    feature_names: list[str],
) -> None:

    final_columns = (
        feature_names
        + [TARGET_COLUMN]
    )

    final_df = labeled_df[
        final_columns
    ].copy()

    final_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        f"\nValidated feedback dataset saved to:"
    )

    print(
        OUTPUT_FILE
    )

    print(
        f"\nRows saved: "
        f"{len(final_df)}"
    )

    print(
        f"Feature columns: "
        f"{len(feature_names)}"
    )

    print(
        f"Target column: "
        f"{TARGET_COLUMN}"
    )


# -------------------------------------------------------
# Main workflow
# -------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("BUILDING VALIDATED FEEDBACK DATASET")
    print("=" * 70)

    feature_names = (
        load_feature_names()
    )

    feedback_df = (
        load_feedback()
    )

    predictions_df = (
        load_prediction_records()
    )

    merged_df = (
        build_validated_feedback_dataset(
            predictions_df=predictions_df,
            feedback_df=feedback_df,
        )
    )

    complete_df = (
        filter_complete_feature_rows(
            merged_df=merged_df,
            feature_names=feature_names,
        )
    )

    labeled_df = (
        add_training_label(
            complete_df
        )
    )

    labeled_df = (
        remove_duplicate_cases(
            labeled_df
        )
    )

    if labeled_df.empty:

        print(
            "\nNo retraining-ready feedback records found."
        )

        print(
            "New prediction records must contain all 30 "
            "features and have Benign/Malignant doctor feedback."
        )

        return

    save_validated_dataset(
        labeled_df=labeled_df,
        feature_names=feature_names,
    )

    print("\n" + "=" * 70)
    print("FEEDBACK DATASET BUILD COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()