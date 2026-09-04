from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import PROJECT_ROOT
from src.promote_model import promote_candidate
from src.retraining_trigger import (
    REAL_FEEDBACK_DATASET,
    SIMULATED_FEEDBACK_DATASET,
    should_retrain,
)


REAL_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_train.csv"
)

REAL_TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_test.csv"
)

CURRENT_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_best_model.pkl"
)

CANDIDATE_MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "retrained_candidate_model.pkl"
)

TARGET_COLUMN = "malignant"
PREDICTION_THRESHOLD = 0.50
RANDOM_STATE = 42

EXPERIMENT_NAME = (
    "Tumor Decision Support - Feedback Retraining"
)


def load_retraining_data(
    feedback_dataset: Path,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:

    if not REAL_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {REAL_TRAIN_PATH}"
        )

    if not REAL_TEST_PATH.exists():
        raise FileNotFoundError(
            f"Test data not found: {REAL_TEST_PATH}"
        )

    if not feedback_dataset.exists():
        raise FileNotFoundError(
            f"Feedback dataset not found: {feedback_dataset}"
        )

    real_train = pd.read_csv(
        REAL_TRAIN_PATH
    )

    real_test = pd.read_csv(
        REAL_TEST_PATH
    )

    feedback_train = pd.read_csv(
        feedback_dataset
    )

    print(
        f"Original training samples : "
        f"{len(real_train)}"
    )

    print(
        f"Validated feedback samples: "
        f"{len(feedback_train)}"
    )

    print(
        f"Untouched test samples     : "
        f"{len(real_test)}"
    )

    return (
        real_train,
        feedback_train,
        real_test,
    )


def validate_feedback_schema(
    real_train: pd.DataFrame,
    feedback_train: pd.DataFrame,
) -> None:

    expected_columns = set(
        real_train.columns
    )

    feedback_columns = set(
        feedback_train.columns
    )

    missing_columns = sorted(
        expected_columns
        - feedback_columns
    )

    extra_columns = sorted(
        feedback_columns
        - expected_columns
    )

    if missing_columns or extra_columns:

        raise ValueError(
            "Feedback dataset schema mismatch. "
            f"Missing={missing_columns}, "
            f"Extra={extra_columns}"
        )


def build_combined_training_data(
    real_train: pd.DataFrame,
    feedback_train: pd.DataFrame,
) -> pd.DataFrame:

    combined_df = pd.concat(
        [
            real_train,
            feedback_train,
        ],
        ignore_index=True,
    )

    before_duplicates = len(
        combined_df
    )

    combined_df = (
        combined_df
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicates_removed = (
        before_duplicates
        - len(combined_df)
    )

    print(
        f"Combined training samples  : "
        f"{len(combined_df)}"
    )

    print(
        f"Duplicate rows removed     : "
        f"{duplicates_removed}"
    )

    return combined_df


def build_model() -> Pipeline:

    base_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    calibrated_svm = (
        CalibratedClassifierCV(
            estimator=base_svm,
            method="sigmoid",
            cv=5,
        )
    )

    return Pipeline(
        [
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                calibrated_svm,
            ),
        ]
    )


def evaluate_model(
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:

    probabilities = (
        model.predict_proba(
            x_test
        )[:, 1]
    )

    predictions = (
        probabilities
        >= PREDICTION_THRESHOLD
    ).astype(int)

    return {
        "accuracy": float(
            accuracy_score(
                y_test,
                predictions,
            )
        ),
        "precision": float(
            precision_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_test,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_test,
                probabilities,
            )
        ),
    }


def print_metrics(
    title: str,
    metrics: dict[str, float],
) -> None:

    print("\n" + title)
    print("-" * 70)

    for name, value in metrics.items():

        print(
            f"{name:<12}: "
            f"{value:.4f}"
        )


def candidate_is_better(
    current_metrics: dict[str, float],
    candidate_metrics: dict[str, float],
) -> bool:

    if (
        candidate_metrics["recall"]
        > current_metrics["recall"]
    ):
        return True

    if (
        candidate_metrics["recall"]
        < current_metrics["recall"]
    ):
        return False

    if (
        candidate_metrics["f1"]
        > current_metrics["f1"]
    ):
        return True

    if (
        candidate_metrics["f1"]
        < current_metrics["f1"]
    ):
        return False

    return (
        candidate_metrics["roc_auc"]
        > current_metrics["roc_auc"]
    )


def retrain_model(
    feedback_dataset: Path,
    test_mode: bool = False,
) -> None:

    print("=" * 70)
    print("FEEDBACK-DRIVEN MODEL RETRAINING")
    print("=" * 70)

    if test_mode:

        print(
            "\nTEST MODE ENABLED"
        )

        print(
            "Using simulated feedback data only."
        )

    else:

        print(
            "\nPRODUCTION MODE"
        )

        print(
            "Using real doctor-validated feedback."
        )

    print(
        f"\nSelected feedback dataset: "
        f"{feedback_dataset}"
    )

    if not should_retrain(
        feedback_dataset
    ):

        print(
            "\nRetraining stopped because "
            "the minimum feedback threshold "
            "has not been reached."
        )

        return

    (
        real_train,
        feedback_train,
        real_test,
    ) = load_retraining_data(
        feedback_dataset
    )

    validate_feedback_schema(
        real_train=real_train,
        feedback_train=feedback_train,
    )

    combined_train = (
        build_combined_training_data(
            real_train=real_train,
            feedback_train=feedback_train,
        )
    )

    feature_columns = [
        column
        for column in real_train.columns
        if column != TARGET_COLUMN
    ]

    x_train = combined_train[
        feature_columns
    ]

    y_train = (
        combined_train[
            TARGET_COLUMN
        ]
        .round()
        .clip(0, 1)
        .astype(int)
    )

    x_test = real_test[
        feature_columns
    ]

    y_test = (
        real_test[
            TARGET_COLUMN
        ]
        .round()
        .clip(0, 1)
        .astype(int)
    )

    if not CURRENT_MODEL_PATH.exists():

        raise FileNotFoundError(
            "Current production model not found: "
            f"{CURRENT_MODEL_PATH}"
        )

    current_model = joblib.load(
        CURRENT_MODEL_PATH
    )

    current_metrics = (
        evaluate_model(
            model=current_model,
            x_test=x_test,
            y_test=y_test,
        )
    )

    candidate_model = (
        build_model()
    )

    print(
        "\nTraining feedback candidate model..."
    )

    candidate_model.fit(
        x_train,
        y_train,
    )

    candidate_metrics = (
        evaluate_model(
            model=candidate_model,
            x_test=x_test,
            y_test=y_test,
        )
    )

    print_metrics(
        "CURRENT PRODUCTION MODEL",
        current_metrics,
    )

    print_metrics(
        "RETRAINED CANDIDATE MODEL",
        candidate_metrics,
    )

    better = candidate_is_better(
        current_metrics=current_metrics,
        candidate_metrics=candidate_metrics,
    )

    joblib.dump(
        candidate_model,
        CANDIDATE_MODEL_PATH,
    )

    print(
        "\nCandidate model saved to:"
    )

    print(
        CANDIDATE_MODEL_PATH
    )

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    run_name = (
        "simulated_feedback_retraining"
        if test_mode
        else "real_feedback_retraining"
    )

    with mlflow.start_run(
        run_name=run_name
    ) as run:

        mlflow.set_tags(
            {
                "pipeline": (
                    "feedback_driven_retraining"
                ),
                "test_mode": (
                    str(test_mode).lower()
                ),
                "feedback_source": (
                    "simulated"
                    if test_mode
                    else "doctor_validated"
                ),
            }
        )

        mlflow.log_params(
            {
                "model": (
                    "Calibrated SVM"
                ),
                "kernel": "rbf",
                "class_weight": "balanced",
                "calibration_method": (
                    "sigmoid"
                ),
                "calibration_cv": 5,
                "prediction_threshold": (
                    PREDICTION_THRESHOLD
                ),
                "original_training_samples": (
                    len(real_train)
                ),
                "feedback_samples": (
                    len(feedback_train)
                ),
                "combined_training_samples": (
                    len(combined_train)
                ),
                "candidate_selected": (
                    better
                ),
            }
        )

        for name, value in (
            current_metrics.items()
        ):

            mlflow.log_metric(
                f"current_{name}",
                value,
            )

        for name, value in (
            candidate_metrics.items()
        ):

            mlflow.log_metric(
                f"candidate_{name}",
                value,
            )

        mlflow.sklearn.log_model(
            sk_model=candidate_model,
            name="candidate_model",
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
        )

        mlflow_run_id = (
            run.info.run_id
        )

        print(
            f"\nMLflow Run ID: "
            f"{mlflow_run_id}"
        )

    if better:

        print(
            "\nCandidate result: "
            "BETTER THAN CURRENT PRODUCTION MODEL"
        )

        if test_mode:

            print(
                "TEST MODE: candidate will NOT "
                "replace the production model."
            )

            promote_candidate(
                mlflow_run_id=mlflow_run_id,
                test_mode=True,
            )

        else:

            print(
                "Candidate passed evaluation."
            )

            print(
                "Starting safe model promotion..."
            )

            promote_candidate(
                mlflow_run_id=mlflow_run_id,
                test_mode=False,
            )

    else:

        print(
            "\nCandidate result: "
            "NOT BETTER THAN CURRENT PRODUCTION MODEL"
        )

        print(
            "Production model remains unchanged."
        )

    print("\n" + "=" * 70)
    print("RETRAINING EVALUATION COMPLETED")
    print("=" * 70)


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Feedback-driven model "
            "retraining workflow"
        )
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help=(
            "Use simulated validated feedback "
            "for pipeline testing."
        ),
    )

    return parser.parse_args()


def main() -> None:

    args = parse_arguments()

    if args.simulate:

        retrain_model(
            feedback_dataset=(
                SIMULATED_FEEDBACK_DATASET
            ),
            test_mode=True,
        )

    else:

        retrain_model(
            feedback_dataset=(
                REAL_FEEDBACK_DATASET
            ),
            test_mode=False,
        )


if __name__ == "__main__":
    main()