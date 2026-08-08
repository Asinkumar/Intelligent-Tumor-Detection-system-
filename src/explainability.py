from __future__ import annotations

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.data import PROJECT_ROOT


MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "final_best_model.pkl"
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

FIGURE_DIR = (
    PROJECT_ROOT
    / "reports"
    / "figures"
)

METRICS_DIR = (
    PROJECT_ROOT
    / "reports"
    / "metrics"
)

TARGET_COLUMN = "malignant"


FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:

    train_df = pd.read_csv(
        REAL_TRAIN_PATH
    )

    test_df = pd.read_csv(
        REAL_TEST_PATH
    )

    feature_names = [
        column
        for column in train_df.columns
        if column != TARGET_COLUMN
    ]

    X_train = train_df[
        feature_names
    ].copy()

    X_test = test_df[
        feature_names
    ].copy()

    y_test = test_df[
        TARGET_COLUMN
    ].copy()

    return X_train, X_test, y_test


def create_explainer(
    model: object,
    background: pd.DataFrame,
) -> shap.Explainer:

    return shap.Explainer(
        model.predict_proba,
        background,
        algorithm="permutation",
    )


def main() -> None:

    print("=" * 70)
    print("PRODUCTION MODEL SHAP EXPLAINABILITY")
    print("=" * 70)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Production model not found: {MODEL_PATH}"
        )

    print(
        f"\nLoading production model:\n{MODEL_PATH}"
    )

    model = joblib.load(
        MODEL_PATH
    )

    X_train, X_test, y_test = load_data()

    print(
        f"Training samples : {len(X_train)}"
    )

    print(
        f"Test samples     : {len(X_test)}"
    )

    print(
        f"Features         : {X_train.shape[1]}"
    )


    # -------------------------------------------------------
    # Keep sample sizes small because permutation SHAP
    # is computationally expensive for calibrated SVM models.
    # -------------------------------------------------------

    background = X_train.sample(
        n=min(
            40,
            len(X_train),
        ),
        random_state=42,
    )

    explain_samples = X_test.sample(
        n=min(
            20,
            len(X_test),
        ),
        random_state=42,
    )


    # -------------------------------------------------------
    # SHAP Explainer
    # -------------------------------------------------------

    print(
        "\nCreating permutation SHAP explainer..."
    )

    explainer = create_explainer(
        model,
        background,
    )


    print(
        "Calculating SHAP values..."
    )

    shap_values = explainer(
        explain_samples
    )


    # predict_proba:
    # class 0 = Benign
    # class 1 = Malignant

    malignant_shap = (
        shap_values[:, :, 1]
    )


    # -------------------------------------------------------
    # Global SHAP Summary Plot
    # -------------------------------------------------------

    print(
        "\nGenerating SHAP summary plot..."
    )

    shap.summary_plot(
        malignant_shap.values,
        explain_samples,
        feature_names=X_train.columns,
        show=False,
    )

    plt.tight_layout()

    summary_path = (
        FIGURE_DIR
        / "shap_summary_plot.png"
    )

    plt.savefig(
        summary_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # -------------------------------------------------------
    # Global SHAP Feature Importance Plot
    # -------------------------------------------------------

    print(
        "Generating SHAP feature importance plot..."
    )

    shap.plots.bar(
        malignant_shap,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    importance_plot_path = (
        FIGURE_DIR
        / "shap_feature_importance.png"
    )

    plt.savefig(
        importance_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # -------------------------------------------------------
    # Numerical Global Importance
    # -------------------------------------------------------

    mean_abs_shap = (
        abs(
            malignant_shap.values
        ).mean(
            axis=0
        )
    )

    importance_df = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Mean_Absolute_SHAP": mean_abs_shap,
        }
    ).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    importance_csv_path = (
        METRICS_DIR
        / "shap_feature_importance.csv"
    )

    importance_df.to_csv(
        importance_csv_path,
        index=False,
    )


    # -------------------------------------------------------
    # Local Explanation
    # -------------------------------------------------------

    print(
        "\nGenerating local SHAP explanation..."
    )

    sample_index = 0

    sample_data = (
        explain_samples
        .iloc[[sample_index]]
        .copy()
    )

    sample_explanation = (
        malignant_shap[
            sample_index
        ]
    )

    shap.plots.waterfall(
        sample_explanation,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    local_plot_path = (
        FIGURE_DIR
        / "shap_local_waterfall.png"
    )

    plt.savefig(
        local_plot_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()


    # -------------------------------------------------------
    # Local Prediction Details
    # -------------------------------------------------------

    malignant_probability = float(
        model.predict_proba(
            sample_data
        )[0][1]
    )

    predicted_label = int(
        malignant_probability >= 0.5
    )

    actual_label = int(
        y_test.loc[
            sample_data.index[0]
        ]
    )


    local_result = pd.DataFrame(
        {
            "Actual_Label": [
                actual_label
            ],
            "Predicted_Label": [
                predicted_label
            ],
            "Malignant_Probability": [
                malignant_probability
            ],
        }
    )

    local_prediction_path = (
        METRICS_DIR
        / "shap_local_prediction.csv"
    )

    local_result.to_csv(
        local_prediction_path,
        index=False,
    )


    # -------------------------------------------------------
    # Top Local Contributing Features
    # -------------------------------------------------------

    local_values = pd.DataFrame(
        {
            "Feature": X_train.columns,
            "Feature_Value": (
                sample_data
                .iloc[0]
                .values
            ),
            "SHAP_Value": (
                sample_explanation.values
            ),
        }
    )

    local_values[
        "Absolute_SHAP"
    ] = (
        local_values[
            "SHAP_Value"
        ].abs()
    )

    local_values = (
        local_values
        .sort_values(
            "Absolute_SHAP",
            ascending=False,
        )
    )

    local_feature_path = (
        METRICS_DIR
        / "shap_local_feature_contributions.csv"
    )

    local_values.to_csv(
        local_feature_path,
        index=False,
    )


    # -------------------------------------------------------
    # Console Output
    # -------------------------------------------------------

    print(
        "\nTop 10 Global SHAP Features"
    )

    print("=" * 70)

    print(
        importance_df
        .head(10)
        .to_string(
            index=False
        )
    )


    print(
        "\nTop 10 Local Contributing Features"
    )

    print("=" * 70)

    print(
        local_values[
            [
                "Feature",
                "Feature_Value",
                "SHAP_Value",
            ]
        ]
        .head(10)
        .to_string(
            index=False
        )
    )


    print(
        "\nLocal Prediction"
    )

    print("=" * 70)

    print(
        f"Actual Label          : "
        f"{'Malignant' if actual_label == 1 else 'Benign'}"
    )

    print(
        f"Predicted Label       : "
        f"{'Malignant' if predicted_label == 1 else 'Benign'}"
    )

    print(
        f"Malignant Probability : "
        f"{malignant_probability:.4f}"
    )


    print(
        "\nSHAP explainability completed successfully."
    )

    print(
        f"\nSummary Plot           : "
        f"{summary_path}"
    )

    print(
        f"Feature Importance Plot: "
        f"{importance_plot_path}"
    )

    print(
        f"Local Waterfall Plot   : "
        f"{local_plot_path}"
    )

    print(
        f"Global Importance CSV  : "
        f"{importance_csv_path}"
    )

    print(
        f"Local Prediction CSV   : "
        f"{local_prediction_path}"
    )

    print(
        f"Local Contributions CSV: "
        f"{local_feature_path}"
    )


if __name__ == "__main__":
    main()