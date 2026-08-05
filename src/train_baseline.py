from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.data import PROJECT_ROOT, load_dataset


FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"
MODEL_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42


def build_models() -> dict[str, object]:
    return {
        "Logistic Regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "K-Nearest Neighbours": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", KNeighborsClassifier(n_neighbors=5)),
            ]
        ),
        "Support Vector Machine": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    SVC(
                        kernel="rbf",
                        probability=True,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            max_depth=5,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def probability_scores(model: object, features: pd.DataFrame) -> object:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    return model.decision_function(features)


def evaluate_model(
    name: str,
    model: object,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float | int | str], object]:
    predictions = model.predict(x_test)
    probabilities = probability_scores(model, x_test)
    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0.0

    result = {
        "model": name,
        "accuracy": accuracy_score(y_test, predictions),
        "malignant_precision": precision_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "malignant_recall": recall_score(
            y_test, predictions, pos_label=1, zero_division=0
        ),
        "f1_score": f1_score(y_test, predictions, pos_label=1, zero_division=0),
        "specificity": specificity,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "false_negatives": int(fn),
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
    }
    return result, probabilities


def run_baselines() -> None:
    sns.set_theme(style="whitegrid")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    features, target = load_dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.20,
        stratify=target,
        random_state=RANDOM_STATE,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results: list[dict[str, float | int | str]] = []
    fitted_models: dict[str, object] = {}
    plt.figure(figsize=(9, 7))

    for name, estimator in build_models().items():
        cv_result = cross_validate(
            clone(estimator),
            x_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )
        model = clone(estimator).fit(x_train, y_train)
        result, probabilities = evaluate_model(name, model, x_test, y_test)
        for metric in scoring:
            result[f"cv_{metric}_mean"] = cv_result[f"test_{metric}"].mean()
            result[f"cv_{metric}_std"] = cv_result[f"test_{metric}"].std()
        results.append(result)
        fitted_models[name] = model

        fpr, tpr, _ = roc_curve(y_test, probabilities)
        plt.plot(fpr, tpr, label=f"{name} (AUC={result['roc_auc']:.3f})")

        display = ConfusionMatrixDisplay.from_predictions(
            y_test,
            model.predict(x_test),
            display_labels=["Benign", "Malignant"],
            cmap="Blues",
            colorbar=False,
        )
        display.ax_.set_title(f"Confusion Matrix – {name}")
        display.figure_.savefig(
            FIGURE_DIR
            / f"confusion_matrix_{name.lower().replace(' ', '_').replace('-', '')}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(display.figure_)

    plt.plot([0, 1], [0, 1], "k--", label="Random classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Baseline Model ROC Curves")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "baseline_roc_curves.png", dpi=200)
    plt.close()

    result_frame = pd.DataFrame(results).sort_values(
        ["malignant_recall", "roc_auc", "f1_score"],
        ascending=False,
    )
    result_frame.to_csv(METRICS_DIR / "baseline_model_results.csv", index=False)

    best_model_name = str(result_frame.iloc[0]["model"])
    joblib.dump(fitted_models[best_model_name], MODEL_DIR / "baseline_best_model.joblib")

    split_report = pd.DataFrame(
        {
            "split": ["training", "testing"],
            "records": [len(x_train), len(x_test)],
            "malignant": [int(y_train.sum()), int(y_test.sum())],
            "benign": [
                int((y_train == 0).sum()),
                int((y_test == 0).sum()),
            ],
        }
    )
    split_report.to_csv(METRICS_DIR / "train_test_split_report.csv", index=False)

    print(result_frame.to_string(index=False))
    print(f"\nSelected baseline model: {best_model_name}")
    print(f"Reports saved to: {METRICS_DIR}")
    print(f"Figures saved to: {FIGURE_DIR}")


if __name__ == "__main__":
    run_baselines()

