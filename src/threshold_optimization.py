import joblib
import numpy as np
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


MODEL_PATH = PROJECT_ROOT / "models" / "final_best_model.pkl"

METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
METRICS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def evaluate_threshold(
    threshold,
    y_true,
    probabilities,
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    ).ravel()

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(
            y_true,
            predictions,
        ),
        "Precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "Specificity": specificity,
        "ROC_AUC": roc_auc_score(
            y_true,
            probabilities,
        ),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
        "True_Positives": int(tp),
        "True_Negatives": int(tn),
    }


def main():

    print("=" * 75)
    print("THRESHOLD OPTIMIZATION")
    print("=" * 75)

    # -------------------------------------------------------
    # Load final model
    # -------------------------------------------------------

    model = joblib.load(
        MODEL_PATH
    )

    # -------------------------------------------------------
    # Load same dataset and test split
    # -------------------------------------------------------

    X, y = load_dataset()

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(
        f"\nTest samples: "
        f"{len(X_test)}"
    )

    # -------------------------------------------------------
    # Get malignant probabilities
    # -------------------------------------------------------

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    # -------------------------------------------------------
    # Test multiple thresholds
    # -------------------------------------------------------

    thresholds = np.arange(
        0.20,
        0.81,
        0.05,
    )

    results = []

    for threshold in thresholds:

        result = evaluate_threshold(
            threshold,
            y_test,
            probabilities,
        )

        results.append(result)

    result_df = pd.DataFrame(
        results
    )

    # -------------------------------------------------------
    # Save all threshold results
    # -------------------------------------------------------

    output_path = (
        METRICS_DIR
        / "threshold_optimization_results.csv"
    )

    result_df.to_csv(
        output_path,
        index=False,
    )

    # -------------------------------------------------------
    # Select recommended threshold
    #
    # Healthcare priority:
    # 1. Recall
    # 2. Minimize false negatives
    # 3. Keep specificity reasonably high
    # 4. F1 score
    #
    # Here we require at least 85% specificity.
    # -------------------------------------------------------

    eligible = result_df[
        result_df["Specificity"] >= 0.85
    ].copy()

    if eligible.empty:

        recommended = result_df.sort_values(
            by=[
                "Recall",
                "False_Negatives",
                "F1",
            ],
            ascending=[
                False,
                True,
                False,
            ],
        ).iloc[0]

    else:

        recommended = eligible.sort_values(
            by=[
                "Recall",
                "False_Negatives",
                "F1",
                "Specificity",
            ],
            ascending=[
                False,
                True,
                False,
                False,
            ],
        ).iloc[0]

    # -------------------------------------------------------
    # Print full table
    # -------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("THRESHOLD COMPARISON")
    print("=" * 75)

    display_columns = [
        "Threshold",
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "Specificity",
        "False_Positives",
        "False_Negatives",
    ]

    print(
        result_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    # -------------------------------------------------------
    # Recommended threshold
    # -------------------------------------------------------

    print("\n")
    print("=" * 75)
    print("RECOMMENDED THRESHOLD")
    print("=" * 75)

    print(
        f"Threshold: "
        f"{recommended['Threshold']:.2f}"
    )

    print(
        f"Accuracy: "
        f"{recommended['Accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{recommended['Precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{recommended['Recall']:.4f}"
    )

    print(
        f"Specificity: "
        f"{recommended['Specificity']:.4f}"
    )

    print(
        f"False Positives: "
        f"{int(recommended['False_Positives'])}"
    )

    print(
        f"False Negatives: "
        f"{int(recommended['False_Negatives'])}"
    )

    # -------------------------------------------------------
    # Save recommended threshold
    # -------------------------------------------------------

    recommended_df = pd.DataFrame(
        [recommended]
    )

    recommended_path = (
        METRICS_DIR
        / "recommended_threshold.csv"
    )

    recommended_df.to_csv(
        recommended_path,
        index=False,
    )

    print(
        f"\nAll Threshold Results: "
        f"{output_path}"
    )

    print(
        f"Recommended Threshold: "
        f"{recommended_path}"
    )


if __name__ == "__main__":
    main()