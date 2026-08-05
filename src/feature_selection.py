from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from src.data import PROJECT_ROOT

# Paths
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "combined_wdbc.csv"

REPORT_DIR = PROJECT_ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"
METRIC_DIR = REPORT_DIR / "metrics"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
METRIC_DIR.mkdir(parents=True, exist_ok=True)


def main():

    print("=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=["malignant"])
    y = df["malignant"]

    print()
    print("Training Random Forest...")
    print()

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42
    )

    model.fit(X, y)

    importance = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values(
        by="Importance",
        ascending=False
    )

    print("=" * 60)
    print("Top 10 Important Features")
    print("=" * 60)

    print(importance.head(10))

    csv_path = METRIC_DIR / "feature_importance.csv"
    importance.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance["Feature"][:15],
        importance["Importance"][:15]
    )

    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 15 Feature Importance")
    plt.gca().invert_yaxis()

    plt.tight_layout()

    fig_path = FIGURE_DIR / "feature_importance.png"

    plt.savefig(fig_path)

    print()
    print("=" * 60)
    print("Feature Importance Saved Successfully")
    print("=" * 60)
    print(f"CSV : {csv_path}")
    print(f"Figure : {fig_path}")


if __name__ == "__main__":
    main()