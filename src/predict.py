from __future__ import annotations

import uuid
from pathlib import Path

import joblib
import pandas as pd

from src.audit import log_event
from src.data import PROJECT_ROOT


MODEL_PATH = PROJECT_ROOT / "models" / "final_best_model.pkl"
THRESHOLD_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_prediction_threshold.txt"
)
REAL_TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "real_train.csv"
)

REPORT_DIR = PROJECT_ROOT / "reports" / "predictions"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COLUMN = "malignant"


def load_model() -> object:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError(
            f"Prediction threshold not found: {THRESHOLD_PATH}"
        )

    threshold = float(
        THRESHOLD_PATH.read_text(encoding="utf-8").strip()
    )

    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "Prediction threshold must be between 0 and 1."
        )

    return threshold


def load_feature_names() -> list[str]:
    if not REAL_TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training data not found: {REAL_TRAIN_PATH}"
        )

    columns = pd.read_csv(
        REAL_TRAIN_PATH,
        nrows=0,
    ).columns.tolist()

    return [
        column
        for column in columns
        if column != TARGET_COLUMN
    ]


MODEL = load_model()
THRESHOLD = load_threshold()
FEATURE_NAMES = load_feature_names()


def generate_case_id() -> str:
    return f"CASE-{uuid.uuid4().hex[:8].upper()}"


def assign_risk_level(
    malignant_probability: float,
) -> str:
    if malignant_probability >= 0.75:
        return "High"

    if malignant_probability >= THRESHOLD:
        return "Moderate"

    return "Low"


def predict_from_features(
    features: list[float],
) -> dict[str, str | float]:
    """
    Predict malignancy using exactly 30 WDBC feature values.

    This is an academic decision-support prototype and is
    not clinically validated.
    """

    if len(features) != len(FEATURE_NAMES):
        raise ValueError(
            f"Exactly {len(FEATURE_NAMES)} "
            "input features are required."
        )

    sample = pd.DataFrame(
        [features],
        columns=FEATURE_NAMES,
    )

    malignant_probability = float(
        MODEL.predict_proba(sample)[0][1]
    )

    prediction = (
        "Malignant"
        if malignant_probability >= THRESHOLD
        else "Benign"
    )

    risk_level = assign_risk_level(
        malignant_probability
    )

    case_id = generate_case_id()

    log_event(
        case_id,
        "PREDICTION_CREATED",
        (
            f"Prediction={prediction}; "
            f"Probability={malignant_probability:.4f}; "
            f"Threshold={THRESHOLD:.2f}; "
            f"Risk={risk_level}"
        ),
    )

    return {
        "case_id": case_id,
        "prediction": prediction,
        "malignant_probability": round(
            malignant_probability * 100,
            2,
        ),
        "decision_threshold": THRESHOLD,
        "risk_level": risk_level,
        "disclaimer": (
            "Academic clinical decision-support prototype; "
            "not a substitute for professional medical diagnosis."
        ),
    }


def main() -> None:
    real_test_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "real_test.csv"
    )

    test_data = pd.read_csv(real_test_path)

    sample = test_data.iloc[0]

    features = sample[FEATURE_NAMES].tolist()
    actual_label = int(sample[TARGET_COLUMN])

    result = predict_from_features(features)

    actual_diagnosis = (
        "Malignant"
        if actual_label == 1
        else "Benign"
    )

    prediction_df = pd.DataFrame(
        {
            "Case_ID": [result["case_id"]],
            "Actual_Diagnosis": [actual_diagnosis],
            "Predicted_Diagnosis": [
                result["prediction"]
            ],
            "Malignant_Probability_Percent": [
                result["malignant_probability"]
            ],
            "Threshold": [
                result["decision_threshold"]
            ],
            "Risk_Level": [
                result["risk_level"]
            ],
        }
    )

    output_path = (
        REPORT_DIR
        / f"{result['case_id']}_prediction.csv"
    )

    prediction_df.to_csv(
        output_path,
        index=False,
    )

    print("=" * 70)
    print("BREAST CANCER RISK PREDICTION")
    print("=" * 70)
    print(f"Case ID              : {result['case_id']}")
    print(f"Actual diagnosis     : {actual_diagnosis}")
    print(f"Predicted diagnosis  : {result['prediction']}")
    print(
        "Malignant probability: "
        f"{result['malignant_probability']:.2f}%"
    )
    print(
        f"Decision threshold   : "
        f"{result['decision_threshold']}"
    )
    print(f"Risk level           : {result['risk_level']}")
    print(f"Prediction saved to  : {output_path}")
    print(f"\n{result['disclaimer']}")


if __name__ == "__main__":
    main()