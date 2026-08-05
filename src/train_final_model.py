from __future__ import annotations

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.models import infer_signature
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.data import PROJECT_ROOT


MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

REAL_TRAIN_PATH = (
    PROJECT_ROOT / "data" / "processed" / "real_train.csv"
)
REAL_TEST_PATH = (
    PROJECT_ROOT / "data" / "processed" / "real_test.csv"
)

MODEL_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
SELECTED_THRESHOLD = 0.50

EXPERIMENT_NAME = "Intelligent Tumor Decision Support System"
REGISTERED_MODEL_NAME = "TumorDecisionSupportSVM"

mlflow.set_experiment(EXPERIMENT_NAME)


def load_final_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    real_train = pd.read_csv(REAL_TRAIN_PATH)
    real_test = pd.read_csv(REAL_TEST_PATH)

    target_column = "malignant"

    if target_column not in real_train.columns:
        raise ValueError(
            f"'{target_column}' column missing in real_train.csv"
        )

    if target_column not in real_test.columns:
        raise ValueError(
            f"'{target_column}' column missing in real_test.csv"
        )

    real_train[target_column] = (
        real_train[target_column]
        .round()
        .clip(0, 1)
        .astype(int)
    )

    real_test[target_column] = (
        real_test[target_column]
        .round()
        .clip(0, 1)
        .astype(int)
    )

    feature_columns = [
        column
        for column in real_train.columns
        if column != target_column
    ]

    missing_columns = (
        set(feature_columns) - set(real_test.columns)
    )

    if missing_columns:
        raise ValueError(
            "Real test dataset missing columns: "
            f"{sorted(missing_columns)}"
        )

    x_train = real_train[feature_columns]
    y_train = real_train[target_column]

    x_test = real_test[feature_columns]
    y_test = real_test[target_column]

    return x_train, x_test, y_train, y_test


def main() -> None:
    print("=" * 70)
    print("TRAINING SELECTED FINAL MODEL")
    print("=" * 70)

    x_train, x_test, y_train, y_test = load_final_data()

    print(f"\nReal training samples: {len(x_train)}")
    print(f"Real testing samples : {len(x_test)}")
    print(f"Input features       : {x_train.shape[1]}")
    print(f"Selected threshold   : {SELECTED_THRESHOLD}")

    with mlflow.start_run(
        run_name="selected_real_only_calibrated_svm"
    ) as run:

        mlflow.set_tags(
            {
                "project": EXPERIMENT_NAME,
                "task": "binary_classification",
                "dataset": (
                    "Wisconsin Diagnostic Breast Cancer"
                ),
                "positive_class": "malignant",
                "selected_model": (
                    "real_only_calibrated_svm"
                ),
                "augmentation_evaluated": "CTGAN",
                "augmentation_selected": "false",
                "selection_metric": "validation_f2",
                "evaluation_data": (
                    "untouched_real_test_set"
                ),
                "data_leakage_prevention": "enabled",
                "clinical_status": (
                    "research_prototype_not_"
                    "clinically_validated"
                ),
            }
        )

        mlflow.log_params(
            {
                "model": "Real-Only Calibrated SVM",
                "kernel": "rbf",
                "class_weight": "balanced",
                "calibration_method": "sigmoid",
                "calibration_cv": 5,
                "random_state": RANDOM_STATE,
                "selected_threshold": SELECTED_THRESHOLD,
                "real_training_samples": len(x_train),
                "real_testing_samples": len(x_test),
                "input_features": x_train.shape[1],
            }
        )

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

        final_model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", calibrated_svm),
            ]
        )

        print("\nTraining selected real-only calibrated SVM...")

        final_model.fit(x_train, y_train)

        probabilities = final_model.predict_proba(
            x_test
        )[:, 1]

        predictions = (
            probabilities >= SELECTED_THRESHOLD
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

        accuracy = accuracy_score(
            y_test,
            predictions,
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

        f1 = f1_score(
            y_test,
            predictions,
            pos_label=1,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y_test,
            probabilities,
        )

        results = {
            "Model": "Real-Only Calibrated SVM",
            "Selected_Threshold": SELECTED_THRESHOLD,
            "Accuracy": accuracy,
            "Malignant_Precision": precision,
            "Malignant_Recall": recall,
            "F1": f1,
            "Specificity": specificity,
            "ROC_AUC": roc_auc,
            "True_Negatives": int(tn),
            "False_Positives": int(fp),
            "False_Negatives": int(fn),
            "True_Positives": int(tp),
            "Real_Training_Samples": len(x_train),
            "Real_Testing_Samples": len(x_test),
        }

        result_df = pd.DataFrame([results])

        model_path = (
            MODEL_DIR
            / "final_best_model.pkl"
        )

        threshold_path = (
            MODEL_DIR
            / "final_prediction_threshold.txt"
        )

        metrics_path = (
            METRICS_DIR
            / "final_model_metrics.csv"
        )

        joblib.dump(
            final_model,
            model_path,
        )

        threshold_path.write_text(
            str(SELECTED_THRESHOLD),
            encoding="utf-8",
        )

        result_df.to_csv(
            metrics_path,
            index=False,
        )

        mlflow.log_metrics(
            {
                "accuracy": float(accuracy),
                "malignant_precision": float(precision),
                "malignant_recall": float(recall),
                "f1_score": float(f1),
                "specificity": float(specificity),
                "roc_auc": float(roc_auc),
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            }
        )

        mlflow.log_artifact(
            str(metrics_path),
            artifact_path="evaluation",
        )

        mlflow.log_artifact(
            str(threshold_path),
            artifact_path="configuration",
        )

        input_example = x_test.iloc[:3].copy()

        signature = infer_signature(
            input_example,
            final_model.predict_proba(
                input_example
            )[:, 1],
        )

        model_info = mlflow.sklearn.log_model(
            sk_model=final_model,
            name="final_model",
            signature=signature,
            input_example=input_example,
            registered_model_name=REGISTERED_MODEL_NAME,
            serialization_format=(
                mlflow.sklearn
                .SERIALIZATION_FORMAT_CLOUDPICKLE
            ),
            tags={
                "model_family": (
                    "calibrated_support_vector_machine"
                ),
                "use_case": "tumor_decision_support",
                "training_pipeline": "real_only",
                "evaluation_pipeline": (
                    "untouched_real_test_data"
                ),
                "prediction_threshold": (
                    str(SELECTED_THRESHOLD)
                ),
            },
        )

        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(
            f"MLflow Model URI: "
            f"{model_info.model_uri}"
        )

    print("\n" + "=" * 70)
    print("SELECTED FINAL MODEL RESULTS")
    print("=" * 70)

    print(result_df.to_string(index=False))

    print("\nConfusion Matrix")
    print("=" * 70)

    print(f"True Negatives  : {tn}")
    print(f"False Positives : {fp}")
    print(f"False Negatives : {fn}")
    print(f"True Positives  : {tp}")

    print(
        "\nSelected final model training "
        "completed successfully."
    )

    print(f"\nFinal model file: {model_path}")
    print(f"Threshold file: {threshold_path}")
    print(f"Metrics file: {metrics_path}")
    print(
        f"Registered MLflow model: "
        f"{REGISTERED_MODEL_NAME}"
    )


if __name__ == "__main__":
    main()