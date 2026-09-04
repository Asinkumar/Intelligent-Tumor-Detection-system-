from __future__ import annotations

import uuid

import joblib
import pandas as pd
import shap

from src.audit import log_event
from src.data import PROJECT_ROOT


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_best_model.pkl"
)

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

REPORT_DIR = (
    PROJECT_ROOT
    / "reports"
    / "predictions"
)

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

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
        THRESHOLD_PATH
        .read_text(
            encoding="utf-8"
        )
        .strip()
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


def load_background_data(
    feature_names: list[str],
) -> pd.DataFrame:

    train_df = pd.read_csv(
        REAL_TRAIN_PATH
    )

    background = (
        train_df[
            feature_names
        ]
        .sample(
            n=min(
                20,
                len(train_df),
            ),
            random_state=42,
        )
        .copy()
    )

    return background


MODEL = load_model()
THRESHOLD = load_threshold()
FEATURE_NAMES = load_feature_names()

BACKGROUND_DATA = load_background_data(
    FEATURE_NAMES
)

SHAP_EXPLAINER = shap.Explainer(
    MODEL.predict_proba,
    BACKGROUND_DATA,
    algorithm="permutation",
)


def generate_case_id() -> str:

    return (
        f"CASE-"
        f"{uuid.uuid4().hex[:8].upper()}"
    )


def assign_risk_level(
    malignant_probability: float,
) -> str:

    if malignant_probability >= 0.75:
        return "High"

    if malignant_probability >= THRESHOLD:
        return "Moderate"

    return "Low"


def explain_prediction(
    sample: pd.DataFrame,
    top_n: int = 5,
) -> list[dict[str, str | float]]:

    shap_values = SHAP_EXPLAINER(
        sample
    )

    malignant_shap = (
        shap_values[
            0,
            :,
            1,
        ]
    )

    explanation_df = pd.DataFrame(
        {
            "feature": FEATURE_NAMES,
            "value": (
                sample
                .iloc[0]
                .values
            ),
            "shap_value": (
                malignant_shap.values
            ),
        }
    )

    explanation_df[
        "absolute_shap"
    ] = (
        explanation_df[
            "shap_value"
        ].abs()
    )

    explanation_df = (
        explanation_df
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .head(top_n)
    )

    factors = []

    for _, row in explanation_df.iterrows():

        shap_value = float(
            row["shap_value"]
        )

        direction = (
            "increases malignant risk"
            if shap_value > 0
            else "reduces malignant risk"
        )

        factors.append(
            {
                "feature": str(
                    row["feature"]
                ),
                "value": round(
                    float(
                        row["value"]
                    ),
                    5,
                ),
                "shap_value": round(
                    shap_value,
                    5,
                ),
                "direction": direction,
            }
        )

    return factors


def save_prediction_record(
    case_id: str,
    prediction: str,
    malignant_probability: float,
    threshold: float,
    risk_level: str,
    sample: pd.DataFrame,
) -> str:

    # Keep the original 30 input features.
    prediction_record = sample.copy()

    # Add Case ID as the first column.
    prediction_record.insert(
        0,
        "Case_ID",
        case_id,
    )

    # Add prediction metadata.
    prediction_record[
        "Predicted_Diagnosis"
    ] = prediction

    prediction_record[
        "Malignant_Probability"
    ] = malignant_probability

    prediction_record[
        "Threshold"
    ] = threshold

    prediction_record[
        "Risk_Level"
    ] = risk_level

    output_path = (
        REPORT_DIR
        / f"{case_id}_prediction.csv"
    )

    prediction_record.to_csv(
        output_path,
        index=False,
    )

    return str(output_path)


def predict_from_features(
    features: list[float],
) -> dict:

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
        MODEL.predict_proba(
            sample
        )[0][1]
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

    top_factors = explain_prediction(
        sample=sample,
        top_n=5,
    )

    prediction_file = save_prediction_record(
        case_id=case_id,
        prediction=prediction,
        malignant_probability=malignant_probability,
        threshold=THRESHOLD,
        risk_level=risk_level,
        sample=sample,
    )

    log_event(
        case_id,
        "PREDICTION_CREATED",
        (
            f"Prediction={prediction}; "
            f"Probability="
            f"{malignant_probability:.4f}; "
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
        "top_factors": top_factors,
        "prediction_file": prediction_file,
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

    test_data = pd.read_csv(
        real_test_path
    )

    sample = test_data.iloc[0]

    features = (
        sample[
            FEATURE_NAMES
        ]
        .tolist()
    )

    actual_label = int(
        sample[
            TARGET_COLUMN
        ]
    )

    result = predict_from_features(
        features
    )

    actual_diagnosis = (
        "Malignant"
        if actual_label == 1
        else "Benign"
    )

    print("=" * 70)
    print("BREAST CANCER RISK PREDICTION")
    print("=" * 70)

    print(
        f"Case ID              : "
        f"{result['case_id']}"
    )

    print(
        f"Actual diagnosis     : "
        f"{actual_diagnosis}"
    )

    print(
        f"Predicted diagnosis  : "
        f"{result['prediction']}"
    )

    print(
        "Malignant probability: "
        f"{result['malignant_probability']:.2f}%"
    )

    print(
        f"Decision threshold   : "
        f"{result['decision_threshold']}"
    )

    print(
        f"Risk level           : "
        f"{result['risk_level']}"
    )

    print(
        "\nTop contributing features"
    )

    print("-" * 70)

    for factor in result[
        "top_factors"
    ]:

        print(
            f"{factor['feature']}: "
            f"value={factor['value']}, "
            f"SHAP={factor['shap_value']}, "
            f"{factor['direction']}"
        )

    print(
        f"\nPrediction saved to  : "
        f"{result['prediction_file']}"
    )

    print(
        f"\n{result['disclaimer']}"
    )


if __name__ == "__main__":
    main()