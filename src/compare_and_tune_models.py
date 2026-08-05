from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sdv.metadata import Metadata
from sdv.single_table import CTGANSynthesizer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import PROJECT_ROOT


REAL_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "real_train.csv"
)
REAL_TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "real_test.csv"
)
FULL_SYNTHETIC_PATH = (
    PROJECT_ROOT / "data" / "synthetic" / "synthetic_wdbc.csv"
)

MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
TEMP_SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"

TARGET_COLUMN = "malignant"
RANDOM_STATE = 42
VALIDATION_SIZE = 0.20
N_SYNTHETIC = 1000
CTGAN_EPOCHS = 300

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
TEMP_SYNTHETIC_DIR.mkdir(parents=True, exist_ok=True)


def build_model() -> Pipeline:
    """
    Build a calibrated SVM pipeline.

    StandardScaler is fitted only inside the training pipeline.
    """

    base_svm = SVC(
        kernel="rbf",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    calibrated_svm = CalibratedClassifierCV(
        estimator=base_svm,
        method="sigmoid",
        cv=5,
    )

    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", calibrated_svm),
        ]
    )


def clean_target(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the target column into binary integer values.
    """

    cleaned = frame.copy()

    cleaned[TARGET_COLUMN] = (
        cleaned[TARGET_COLUMN]
        .round()
        .clip(0, 1)
        .astype(int)
    )

    return cleaned


def generate_validation_synthetic_data(
    real_model_train: pd.DataFrame,
) -> pd.DataFrame:
    """
    Generate synthetic data using only the model-training portion.

    The validation and final test records are never shown to this CTGAN.
    """

    print("\nTraining validation CTGAN using model-training records only...")

    metadata = Metadata.detect_from_dataframe(
        real_model_train
    )

    synthesizer = CTGANSynthesizer(
        metadata=metadata,
        epochs=CTGAN_EPOCHS,
        verbose=True,
    )

    synthesizer.fit(real_model_train)

    print(
        f"\nGenerating {N_SYNTHETIC} validation-stage "
        "synthetic records..."
    )

    synthetic = synthesizer.sample(
        num_rows=N_SYNTHETIC
    )

    synthetic = clean_target(synthetic)

    validation_synthetic_path = (
        TEMP_SYNTHETIC_DIR
        / "validation_train_only_synthetic.csv"
    )

    synthetic.to_csv(
        validation_synthetic_path,
        index=False,
    )

    print(
        "Validation-stage synthetic data saved at: "
        f"{validation_synthetic_path}"
    )

    return synthetic


def f_beta_score(
    precision: float,
    recall: float,
    beta: float = 2.0,
) -> float:
    """
    Calculate F-beta score.

    Beta=2 gives more importance to recall than precision.
    """

    beta_squared = beta**2

    denominator = (
        beta_squared * precision
        + recall
    )

    if denominator == 0:
        return 0.0

    return (
        (1 + beta_squared)
        * precision
        * recall
        / denominator
    )


def select_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> tuple[float, dict[str, float]]:
    """
    Select a threshold using validation data only.

    The threshold maximising F2 is selected because this
    healthcare prototype prioritises malignant recall.
    """

    best_threshold = 0.50

    best_result = {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "f2": 0.0,
    }

    for threshold in np.arange(
        0.10,
        0.91,
        0.01,
    ):
        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        f2 = f_beta_score(
            precision=precision,
            recall=recall,
            beta=2.0,
        )

        candidate = (
            f2,
            recall,
            precision,
        )

        current_best = (
            best_result["f2"],
            best_result["recall"],
            best_result["precision"],
        )

        if candidate > current_best:
            best_threshold = float(
                round(threshold, 2)
            )

            best_result = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "f2": float(f2),
            }

    return best_threshold, best_result


def evaluate_model(
    model_name: str,
    model: Pipeline,
    x_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float,
    validation_f2: float,
) -> dict[str, float | int | str]:
    """
    Evaluate a fitted model on the untouched real test set.
    """

    probabilities = model.predict_proba(
        x_test
    )[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    precision = precision_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label=1,
        zero_division=0,
    )

    return {
        "model": model_name,
        "selected_threshold": threshold,
        "validation_f2": validation_f2,
        "test_accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "test_malignant_precision": precision,
        "test_malignant_recall": recall,
        "test_f1": f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        ),
        "test_f2": f_beta_score(
            precision=precision,
            recall=recall,
            beta=2.0,
        ),
        "test_specificity": specificity,
        "test_roc_auc": roc_auc_score(
            y_test,
            probabilities,
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def main() -> None:
    print("=" * 75)
    print("REAL-ONLY VS CTGAN-AUGMENTED MODEL COMPARISON")
    print("=" * 75)

    real_train = clean_target(
        pd.read_csv(REAL_TRAIN_PATH)
    )

    real_test = clean_target(
        pd.read_csv(REAL_TEST_PATH)
    )

    full_synthetic = clean_target(
        pd.read_csv(FULL_SYNTHETIC_PATH)
    )

    feature_columns = [
        column
        for column in real_train.columns
        if column != TARGET_COLUMN
    ]

    real_model_train, real_validation = (
        train_test_split(
            real_train,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=real_train[TARGET_COLUMN],
        )
    )

    print(
        f"\nReal model-training records : "
        f"{len(real_model_train)}"
    )
    print(
        f"Real validation records     : "
        f"{len(real_validation)}"
    )
    print(
        f"Untouched real test records : "
        f"{len(real_test)}"
    )

    validation_synthetic = (
        generate_validation_synthetic_data(
            real_model_train
        )
    )

    augmented_model_train = pd.concat(
        [
            real_model_train,
            validation_synthetic,
        ],
        ignore_index=True,
    ).sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    x_real_train = real_model_train[
        feature_columns
    ]
    y_real_train = real_model_train[
        TARGET_COLUMN
    ]

    x_augmented_train = augmented_model_train[
        feature_columns
    ]
    y_augmented_train = augmented_model_train[
        TARGET_COLUMN
    ]

    x_validation = real_validation[
        feature_columns
    ]
    y_validation = real_validation[
        TARGET_COLUMN
    ]

    x_test = real_test[
        feature_columns
    ]
    y_test = real_test[
        TARGET_COLUMN
    ]

    print("\nTraining real-only validation model...")

    real_only_validation_model = build_model()

    real_only_validation_model.fit(
        x_real_train,
        y_real_train,
    )

    real_validation_probabilities = (
        real_only_validation_model.predict_proba(
            x_validation
        )[:, 1]
    )

    (
        real_only_threshold,
        real_only_validation_metrics,
    ) = select_threshold(
        y_validation,
        real_validation_probabilities,
    )

    print(
        "Real-only selected threshold: "
        f"{real_only_threshold}"
    )
    print(
        "Real-only validation recall: "
        f"{real_only_validation_metrics['recall']:.4f}"
    )
    print(
        "Real-only validation F2: "
        f"{real_only_validation_metrics['f2']:.4f}"
    )

    print("\nTraining augmented validation model...")

    augmented_validation_model = build_model()

    augmented_validation_model.fit(
        x_augmented_train,
        y_augmented_train,
    )

    augmented_validation_probabilities = (
        augmented_validation_model.predict_proba(
            x_validation
        )[:, 1]
    )

    (
        augmented_threshold,
        augmented_validation_metrics,
    ) = select_threshold(
        y_validation,
        augmented_validation_probabilities,
    )

    print(
        "Augmented selected threshold: "
        f"{augmented_threshold}"
    )
    print(
        "Augmented validation recall: "
        f"{augmented_validation_metrics['recall']:.4f}"
    )
    print(
        "Augmented validation F2: "
        f"{augmented_validation_metrics['f2']:.4f}"
    )

    # Select the model using validation results only.
    use_augmented_model = (
        augmented_validation_metrics["f2"]
        > real_only_validation_metrics["f2"]
    )

    if (
        augmented_validation_metrics["f2"]
        == real_only_validation_metrics["f2"]
    ):
        use_augmented_model = (
            augmented_validation_metrics["recall"]
            > real_only_validation_metrics["recall"]
        )

    print("\nRefitting both models using all real training data...")

    final_real_only_model = build_model()

    final_real_only_model.fit(
        real_train[feature_columns],
        real_train[TARGET_COLUMN],
    )

    full_augmented_train = pd.concat(
        [
            real_train,
            full_synthetic,
        ],
        ignore_index=True,
    ).sample(
        frac=1,
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    final_augmented_model = build_model()

    final_augmented_model.fit(
        full_augmented_train[feature_columns],
        full_augmented_train[TARGET_COLUMN],
    )

    real_only_result = evaluate_model(
        model_name="Real-Only Calibrated SVM",
        model=final_real_only_model,
        x_test=x_test,
        y_test=y_test,
        threshold=real_only_threshold,
        validation_f2=(
            real_only_validation_metrics["f2"]
        ),
    )

    augmented_result = evaluate_model(
        model_name="CTGAN-Augmented Calibrated SVM",
        model=final_augmented_model,
        x_test=x_test,
        y_test=y_test,
        threshold=augmented_threshold,
        validation_f2=(
            augmented_validation_metrics["f2"]
        ),
    )

    comparison = pd.DataFrame(
        [
            real_only_result,
            augmented_result,
        ]
    )

    comparison_path = (
        METRICS_DIR
        / "real_vs_augmented_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    if use_augmented_model:
        selected_model_name = (
            "CTGAN-Augmented Calibrated SVM"
        )
        selected_model = final_augmented_model
        selected_threshold = augmented_threshold
    else:
        selected_model_name = (
            "Real-Only Calibrated SVM"
        )
        selected_model = final_real_only_model
        selected_threshold = real_only_threshold

    selected_model_path = (
        MODEL_DIR
        / "final_best_model.pkl"
    )

    joblib.dump(
        selected_model,
        selected_model_path,
    )

    threshold_path = (
        MODEL_DIR
        / "final_prediction_threshold.txt"
    )

    threshold_path.write_text(
        str(selected_threshold),
        encoding="utf-8",
    )

    print("\n" + "=" * 75)
    print("MODEL COMPARISON RESULTS")
    print("=" * 75)

    print(
        comparison.to_string(
            index=False
        )
    )

    print("\nSelected using validation performance:")
    print(f"Model     : {selected_model_name}")
    print(f"Threshold : {selected_threshold}")

    print(
        f"\nComparison saved at: "
        f"{comparison_path}"
    )
    print(
        f"Selected model saved at: "
        f"{selected_model_path}"
    )
    print(
        f"Threshold saved at: "
        f"{threshold_path}"
    )


if __name__ == "__main__":
    main()