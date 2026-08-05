# Explainable Breast Cancer Classification with MLOps

Academic clinical decision-support prototype for classifying Wisconsin
Diagnostic Breast Cancer records as malignant or benign.

## Review I milestone

- Dataset acquisition and quality report
- Exploratory data analysis
- Five baseline classification models
- Healthcare-oriented evaluation, especially malignant recall
- Initial MLOps architecture and implementation plan

## Safety disclaimer

This project is not clinically validated and must not replace diagnosis or
advice from qualified healthcare professionals.

## Local setup

```powershell
$python = "C:\Users\kalaivendhan\anaconda3\python.exe"
& $python -m pip install -r requirements.txt
& $python -m src.eda
& $python -m src.train_baseline
```

Generated evidence is saved under `data/` and `reports/`.

