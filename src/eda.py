from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.data import PROJECT_ROOT, export_raw_dataset, load_dataset


FIGURE_DIR = PROJECT_ROOT / "reports" / "figures"
METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def save_current_figure(name: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / name, dpi=200, bbox_inches="tight")
    plt.close()


def run_eda() -> None:
    sns.set_theme(style="whitegrid")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = export_raw_dataset()
    features, target = load_dataset()
    frame = features.copy()
    frame["diagnosis"] = target.map({1: "Malignant", 0: "Benign"})

    quality_report = pd.DataFrame(
        {
            "metric": [
                "rows",
                "input_features",
                "missing_values",
                "duplicate_rows",
                "malignant_records",
                "benign_records",
            ],
            "value": [
                len(features),
                features.shape[1],
                int(features.isna().sum().sum()),
                int(features.duplicated().sum()),
                int(target.sum()),
                int((target == 0).sum()),
            ],
        }
    )
    quality_report.to_csv(METRICS_DIR / "dataset_quality_report.csv", index=False)
    features.describe().T.to_csv(METRICS_DIR / "descriptive_statistics.csv")

    class_counts = frame["diagnosis"].value_counts()
    ax = sns.barplot(
        x=class_counts.index,
        y=class_counts.values,
        hue=class_counts.index,
        legend=False,
        palette={"Benign": "#4C956C", "Malignant": "#D1495B"},
    )
    ax.set_title("Diagnosis Class Distribution")
    ax.set_xlabel("Diagnosis")
    ax.set_ylabel("Number of records")
    for container in ax.containers:
        ax.bar_label(container)
    save_current_figure("class_distribution.png")

    selected_features = [
        "mean radius",
        "mean texture",
        "mean concavity",
        "mean concave points",
        "worst radius",
        "worst perimeter",
    ]
    melted = frame.melt(
        id_vars="diagnosis",
        value_vars=selected_features,
        var_name="feature",
        value_name="value",
    )
    grid = sns.FacetGrid(
        melted,
        col="feature",
        col_wrap=3,
        hue="diagnosis",
        sharex=False,
        sharey=False,
        height=3,
        palette={"Benign": "#4C956C", "Malignant": "#D1495B"},
    )
    grid.map(sns.kdeplot, "value", fill=False, common_norm=False)
    grid.add_legend()
    grid.set_titles("{col_name}")
    grid.figure.suptitle("Selected Feature Distributions by Diagnosis", y=1.03)
    grid.figure.savefig(
        FIGURE_DIR / "selected_feature_distributions.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(grid.figure)

    plt.figure(figsize=(12, 6))
    box_data = frame.melt(
        id_vars="diagnosis",
        value_vars=[
            "mean radius",
            "mean texture",
            "mean area",
            "worst radius",
            "worst texture",
            "worst area",
        ],
        var_name="feature",
        value_name="value",
    )
    sns.boxplot(
        data=box_data,
        x="feature",
        y="value",
        hue="diagnosis",
        showfliers=True,
        palette={"Benign": "#4C956C", "Malignant": "#D1495B"},
    )
    plt.xticks(rotation=30, ha="right")
    plt.title("Feature Ranges and Potential Outliers")
    save_current_figure("feature_boxplots.png")

    correlation = features.corr()
    plt.figure(figsize=(16, 13))
    sns.heatmap(correlation, cmap="coolwarm", center=0, square=False)
    plt.title("Feature Correlation Heatmap")
    save_current_figure("correlation_heatmap.png")

    target_correlations = (
        features.assign(malignant=target)
        .corr(numeric_only=True)["malignant"]
        .drop("malignant")
        .sort_values(key=abs, ascending=False)
        .rename("correlation_with_malignant")
    )
    target_correlations.to_csv(METRICS_DIR / "target_correlations.csv")

    print(f"Raw dataset: {raw_path}")
    print(quality_report.to_string(index=False))
    print(f"EDA figures: {FIGURE_DIR}")
    print(f"EDA reports: {METRICS_DIR}")


if __name__ == "__main__":
    run_eda()

