# Review I – Actual Experimental Results

## Dataset quality

- Records: 569
- Input features: 30 numerical features
- Benign records: 357
- Malignant records: 212
- Missing values: 0
- Duplicate feature rows: 0

The dataset has a moderate class imbalance: approximately 62.7% benign and
37.3% malignant. A stratified train-test split was therefore used.

## EDA observations

- Radius, perimeter, and area measurements have strong positive correlations.
- Their corresponding worst-value features are also highly correlated.
- Malignant and benign records show visible distribution differences for
  radius, perimeter, concavity, and concave-point measurements.
- Feature ranges differ substantially, supporting the use of standardisation
  for Logistic Regression, KNN, and SVM.
- Potential outliers are retained at this stage because extreme medical
  measurements may be clinically meaningful.

## Baseline experimental design

- Train-test ratio: 80:20
- Split: stratified
- Random state: 42
- Cross-validation: five-fold stratified cross-validation
- Positive class: malignant
- Models: Logistic Regression, KNN, SVM, Decision Tree, and Random Forest

## Test-set results

| Model | Accuracy | Malignant Precision | Malignant Recall | F1 | Specificity | ROC-AUC | False Negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| Support Vector Machine | 98.25% | 97.62% | 97.62% | 97.62% | 98.61% | 99.54% | 1 |
| Logistic Regression | 97.37% | 97.56% | 95.24% | 96.39% | 98.61% | 99.54% | 2 |
| Random Forest | 96.49% | 100.00% | 90.48% | 95.00% | 100.00% | 99.70% | 4 |
| K-Nearest Neighbours | 95.61% | 97.44% | 90.48% | 93.83% | 98.61% | 98.23% | 4 |
| Decision Tree | 92.98% | 92.50% | 88.10% | 90.24% | 95.83% | 89.88% | 5 |

## Initial interpretation

Support Vector Machine is the selected baseline candidate because it achieved
the highest test accuracy and malignant recall, with only one false-negative
prediction. Random Forest produced the highest test ROC-AUC and perfect
malignant precision, but it missed four malignant records. For this healthcare
classification prototype, reducing false negatives is prioritised over using
accuracy or ROC-AUC alone.

These are baseline results from one fixed test split. They are not final
clinical claims. Hyperparameter tuning, threshold analysis, explainability, and
robust validation will be performed in later project stages.

## Generated evidence

- `reports/metrics/dataset_quality_report.csv`
- `reports/metrics/descriptive_statistics.csv`
- `reports/metrics/target_correlations.csv`
- `reports/metrics/baseline_model_results.csv`
- `reports/metrics/train_test_split_report.csv`
- `reports/figures/class_distribution.png`
- `reports/figures/selected_feature_distributions.png`
- `reports/figures/feature_boxplots.png`
- `reports/figures/correlation_heatmap.png`
- `reports/figures/baseline_roc_curves.png`
- Five model-specific confusion-matrix images

