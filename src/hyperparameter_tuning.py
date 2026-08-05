from pathlib import Path
import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.data import load_dataset, PROJECT_ROOT


MODELS_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "metrics"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def evaluate(name, model, X_test, y_test):

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    return {
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred),
        "Recall": recall_score(y_test, pred),
        "F1": f1_score(y_test, pred),
        "ROC_AUC": roc_auc_score(y_test, prob),
    }


def main():

    print("=" * 70)
    print("Loading Dataset")
    print("=" * 70)

    X, y = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    results = []

    #########################################################
    # Logistic Regression
    #########################################################

    print("\nTuning Logistic Regression...")

    lr_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000))
    ])

    lr_params = {
        "model__C": [0.1, 1, 10],
        "model__solver": ["liblinear", "lbfgs"],
    }

    lr_grid = GridSearchCV(
        lr_pipe,
        lr_params,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    lr_grid.fit(X_train, y_train)

    print("Best Params:", lr_grid.best_params_)

    results.append(
        evaluate(
            "Logistic Regression",
            lr_grid.best_estimator_,
            X_test,
            y_test,
        )
    )

    joblib.dump(
        lr_grid.best_estimator_,
        MODELS_DIR / "logistic_tuned.pkl"
    )

    #########################################################
    # Random Forest
    #########################################################

    print("\nTuning Random Forest...")

    rf = RandomForestClassifier(random_state=42)

    rf_params = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    }

    rf_grid = GridSearchCV(
        rf,
        rf_params,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    rf_grid.fit(X_train, y_train)

    print("Best Params:", rf_grid.best_params_)

    results.append(
        evaluate(
            "Random Forest",
            rf_grid.best_estimator_,
            X_test,
            y_test,
        )
    )

    joblib.dump(
        rf_grid.best_estimator_,
        MODELS_DIR / "random_forest_tuned.pkl"
    )

    #########################################################
    # Support Vector Machine
    #########################################################

    print("\nTuning Support Vector Machine...")

    svm_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(probability=True))
    ])

    svm_params = {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"],
    }

    svm_grid = GridSearchCV(
        svm_pipe,
        svm_params,
        cv=5,
        scoring="roc_auc",
        n_jobs=-1,
    )

    svm_grid.fit(X_train, y_train)

    print("Best Params:", svm_grid.best_params_)

    results.append(
        evaluate(
            "Support Vector Machine",
            svm_grid.best_estimator_,
            X_test,
            y_test,
        )
    )

    joblib.dump(
        svm_grid.best_estimator_,
        MODELS_DIR / "svm_tuned.pkl"
    )

    #########################################################

    df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print(df)
    print("=" * 70)

    df.to_csv(
        REPORT_DIR / "hyperparameter_results.csv",
        index=False,
    )

    best = df.sort_values(
        "ROC_AUC",
        ascending=False,
    ).iloc[0]

    print("\nBest Tuned Model")
    print(best)


if __name__ == "__main__":
    main()