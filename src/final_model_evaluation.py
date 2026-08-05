import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from src.data import PROJECT_ROOT, load_dataset


MODEL_DIR = PROJECT_ROOT / "models"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"

METRICS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_model(name, model, X_test, y_test):

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "Model": name,
        "Accuracy": accuracy_score(
            y_test,
            predictions,
        ),
        "Precision": precision_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_test,
            predictions,
            zero_division=0,
        ),
        "Specificity": specificity,
        "ROC_AUC": roc_auc_score(
            y_test,
            probabilities,
        ),
        "False_Negatives": int(fn),
    }


def main():

    print("=" * 75)
    print("FINAL MODEL EVALUATION")
    print("=" * 75)

    # -------------------------------------------------------
    # Load Dataset
    # -------------------------------------------------------

    X, y = load_dataset()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nEvaluation dataset size: "
        f"{len(X_test)} samples"
    )

    # -------------------------------------------------------
    # Load Models
    # -------------------------------------------------------

    models = {
        "Baseline Best Model": joblib.load(
            MODEL_DIR
            / "baseline_best_model.joblib"
        ),

        "Tuned Logistic Regression": joblib.load(
            MODEL_DIR
            / "logistic_tuned.pkl"
        ),

        "Tuned Random Forest": joblib.load(
            MODEL_DIR
            / "random_forest_tuned.pkl"
        ),

        "Tuned SVM": joblib.load(
            MODEL_DIR
            / "svm_tuned.pkl"
        ),
    }

    # -------------------------------------------------------
    # Evaluate Models
    # -------------------------------------------------------

    results = []

    for name, model in models.items():

        print(
            f"\nEvaluating: {name}"
        )

        result = evaluate_model(
            name,
            model,
            X_test,
            y_test,
        )

        results.append(result)

    result_df = pd.DataFrame(
        results
    )

    # -------------------------------------------------------
    # Sort Models
    #
    # Healthcare priority:
    # 1. Recall
    # 2. False Negatives
    # 3. ROC-AUC
    # 4. F1
    # -------------------------------------------------------

    result_df = result_df.sort_values(
        by=[
            "Recall",
            "False_Negatives",
            "ROC_AUC",
            "F1",
        ],
        ascending=[
            False,
            True,
            False,
            False,
        ],
    ).reset_index(drop=True)

    print("\n")
    print("=" * 75)
    print("MODEL COMPARISON RESULTS")
    print("=" * 75)

    print(
        result_df.to_string(
            index=False
        )
    )

    # -------------------------------------------------------
    # Save Results
    # -------------------------------------------------------

    comparison_path = (
        METRICS_DIR
        / "final_model_comparison.csv"
    )

    result_df.to_csv(
        comparison_path,
        index=False,
    )

    # -------------------------------------------------------
    # Select Final Model
    # -------------------------------------------------------

    best_model_name = (
        result_df.iloc[0]["Model"]
    )

    final_model = models[
        best_model_name
    ]

    final_model_path = (
        MODEL_DIR
        / "final_best_model.pkl"
    )

    joblib.dump(
        final_model,
        final_model_path,
    )

    # -------------------------------------------------------
    # Display Final Selection
    # -------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("FINAL SELECTED MODEL")
    print("=" * 75)

    print(
        f"Model: "
        f"{best_model_name}"
    )

    print(
        f"Recall: "
        f"{result_df.iloc[0]['Recall']:.4f}"
    )

    print(
        f"False Negatives: "
        f"{int(result_df.iloc[0]['False_Negatives'])}"
    )

    print(
        f"ROC-AUC: "
        f"{result_df.iloc[0]['ROC_AUC']:.4f}"
    )

    print(
        f"F1 Score: "
        f"{result_df.iloc[0]['F1']:.4f}"
    )

    print(
        f"\nComparison Report: "
        f"{comparison_path}"
    )

    print(
        f"Final Model Saved: "
        f"{final_model_path}"
    )


if __name__ == "__main__":
    main()