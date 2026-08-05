import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import shap

from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT, load_dataset


MODEL_PATH = PROJECT_ROOT / "models" / "svm_tuned.pkl"

FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 70)
    print("Loading Tuned SVM Model")
    print("=" * 70)

    model = joblib.load(MODEL_PATH)

    # Load dataset
    X, y = load_dataset()

    # Create same train-test split used during tuning
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(f"Test samples available: {len(X_test)}")

    # Smaller sample sizes are used because
    # model-agnostic SHAP explanation can be slow.
    background = X_test.sample(
        n=min(50, len(X_test)),
        random_state=42,
    )

    explain_samples = X_test.sample(
        n=min(30, len(X_test)),
        random_state=42,
    )

    # -------------------------------------------------------
    # Create SHAP Explainer
    # -------------------------------------------------------

    print("\nCreating SHAP explainer...")

    explainer = shap.Explainer(
        model.predict_proba,
        background,
        algorithm="permutation",
    )

    print("Calculating SHAP values...")

    shap_values = explainer(explain_samples)

    # predict_proba output:
    # class 0 = Benign
    # class 1 = Malignant
    malignant_shap = shap_values[:, :, 1]

    # -------------------------------------------------------
    # Global SHAP Summary Plot
    # -------------------------------------------------------

    print("\nGenerating SHAP summary plot...")

    shap.summary_plot(
        malignant_shap.values,
        explain_samples,
        feature_names=X.columns,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_summary_plot.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------
    # Global SHAP Feature Importance Plot
    # -------------------------------------------------------

    print("Generating SHAP bar importance plot...")

    shap.plots.bar(
        malignant_shap,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_feature_importance.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------
    # Save Numerical SHAP Feature Importance
    # -------------------------------------------------------

    mean_abs_shap = abs(
        malignant_shap.values
    ).mean(axis=0)

    importance_df = pd.DataFrame(
        {
            "Feature": X.columns,
            "Mean_Absolute_SHAP": mean_abs_shap,
        }
    ).sort_values(
        "Mean_Absolute_SHAP",
        ascending=False,
    )

    importance_df.to_csv(
        METRICS_DIR / "shap_feature_importance.csv",
        index=False,
    )

    # -------------------------------------------------------
    # Local SHAP Explanation
    # -------------------------------------------------------

    print("\nGenerating Local SHAP Explanation...")

    # Select first sample from SHAP sample set
    sample_index = 0

    sample_data = explain_samples.iloc[
        [sample_index]
    ].copy()

    sample_explanation = malignant_shap[
        sample_index
    ]

    # Generate local waterfall plot
    shap.plots.waterfall(
        sample_explanation,
        max_display=15,
        show=False,
    )

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "shap_local_waterfall.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    # -------------------------------------------------------
    # Local Prediction Details
    # -------------------------------------------------------

    prediction = model.predict(
        sample_data
    )[0]

    probability = model.predict_proba(
        sample_data
    )[0][1]

    actual_label = y_test.loc[
        sample_data.index[0]
    ]

    local_result = pd.DataFrame(
        {
            "Actual_Label": [
                int(actual_label)
            ],
            "Predicted_Label": [
                int(prediction)
            ],
            "Malignant_Probability": [
                float(probability)
            ],
        }
    )

    local_result.to_csv(
        METRICS_DIR / "shap_local_prediction.csv",
        index=False,
    )

    # -------------------------------------------------------
    # Display Results
    # -------------------------------------------------------

    print("\nTop 10 SHAP Features")
    print("=" * 70)

    print(
        importance_df
        .head(10)
        .to_string(index=False)
    )

    print("\nLocal Prediction")
    print("=" * 70)

    print(
        f"Actual Label: "
        f"{'Malignant' if actual_label == 1 else 'Benign'}"
    )

    print(
        f"Predicted Label: "
        f"{'Malignant' if prediction == 1 else 'Benign'}"
    )

    print(
        f"Malignant Probability: "
        f"{probability:.4f}"
    )

    print("\nSHAP Explainability Completed Successfully")

    print(
        f"\nSummary Plot: "
        f"{FIGURE_DIR / 'shap_summary_plot.png'}"
    )

    print(
        f"Feature Importance Plot: "
        f"{FIGURE_DIR / 'shap_feature_importance.png'}"
    )

    print(
        f"Local Waterfall Plot: "
        f"{FIGURE_DIR / 'shap_local_waterfall.png'}"
    )

    print(
        f"SHAP Importance CSV: "
        f"{METRICS_DIR / 'shap_feature_importance.csv'}"
    )

    print(
        f"Local Prediction CSV: "
        f"{METRICS_DIR / 'shap_local_prediction.csv'}"
    )


if __name__ == "__main__":
    main()