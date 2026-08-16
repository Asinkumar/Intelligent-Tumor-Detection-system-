# Intelligent Tumor Decision Support System

An end-to-end Explainable AI and MLOps-based clinical decision-support prototype for breast cancer risk classification using the Wisconsin Diagnostic Breast Cancer dataset.

The system predicts whether a tumor record is **Malignant** or **Benign**, estimates malignant probability, assigns a risk level, and provides feature-level explanations for the prediction.

> **Academic Prototype:** This system is intended for research and educational purposes only. It is not clinically validated and must not replace professional medical diagnosis, pathology, imaging, biopsy, or medical advice.

## Live Application

### Frontend
GitHub Pages:

https://asinkumar.github.io/Intelligent-Tumor-Detection-system-/

### Backend API
Render:

https://tumor-decision-support.onrender.com/

### API Health Check
https://tumor-decision-support.onrender.com/health

### Interactive API Documentation
https://tumor-decision-support.onrender.com/docs

---

## Key Features

- Breast tumor classification: Malignant / Benign
- Malignant probability estimation
- Risk-level assessment
- 30 diagnostic tumor features
- CSV upload
- Sample patient case
- Manual feature entry
- Explainable AI using SHAP
- Top contributing features for each prediction
- REST API using FastAPI
- Dockerized backend
- DVC-based model artifact management
- Google Drive DVC remote storage
- GitHub Actions CI/CD
- Docker Hub image deployment
- Render cloud deployment
- GitHub Pages frontend deployment
- Automated testing and validation

---

## System Architecture

```text
User
  |
  v
GitHub Pages Frontend
  |
  | HTTPS / JSON
  v
FastAPI Backend - Render
  |
  v
Preprocessing + ML Model
  |
  +----> Prediction
  |
  +----> Malignant Probability
  |
  +----> Risk Level
  |
  +----> SHAP Explainability
  |
  v
JSON Response
  |
  v
Frontend AI Assessment
```

---

## Input Features

The model uses the 30 diagnostic features from the Wisconsin Diagnostic Breast Cancer dataset.

They are grouped into:

- Mean Measurements
- Measurement Errors
- Worst Measurements

Examples include:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Compactness
- Concavity
- Concave Points
- Symmetry
- Fractal Dimension

---

## Explainable AI

The application provides prediction-level explainability using **SHAP**.

For every analyzed case, the system identifies the tumor characteristics with the strongest influence on the model prediction.

The interface indicates whether each important feature:

- increases malignant risk, or
- reduces malignant risk.

This improves transparency compared with displaying only the final classification.

---

## MLOps Pipeline

The project implements an end-to-end MLOps workflow.

```text
Dataset
   |
   v
EDA & Data Validation
   |
   v
Model Training
   |
   v
Model Evaluation
   |
   v
Model Selection
   |
   v
DVC Artifact Versioning
   |
   v
GitHub Repository
   |
   v
GitHub Actions CI/CD
   |
   +----> Pull model artifacts from DVC
   |
   +----> Build Docker Image
   |
   +----> Push Docker Image
   |
   v
Render Deployment
   |
   v
Production FastAPI Service
```

---

## Technology Stack

### Machine Learning
- Python
- Scikit-learn
- Pandas
- NumPy
- SHAP

### Backend
- FastAPI
- Uvicorn

### Frontend
- HTML
- CSS
- JavaScript

### MLOps / DevOps
- Git
- GitHub
- GitHub Actions
- DVC
- Google Drive
- Docker
- Docker Hub
- Render
- GitHub Pages

---

## API Endpoints

### Root

```http
GET /
```

Returns basic project information.

### Health Check

```http
GET /health
```

Used to verify that the deployed API is operational.

### Prediction

```http
POST /predict
```

Expected request format:

```json
{
  "features": [
    11.41,
    10.82,
    73.34
  ]
}
```

The complete request contains all **30 diagnostic features**.

Example response structure:

```json
{
  "case_id": "CASE-XXXXXXXX",
  "prediction": "Benign",
  "malignant_probability": 0.0083,
  "decision_threshold": 0.5,
  "risk_level": "Low",
  "top_factors": [],
  "disclaimer": "Academic clinical decision-support prototype"
}
```

---

## Local Development

### Create and activate virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Run the API

```powershell
uvicorn app.main:app --reload
```

---

## DVC Model Artifacts

Large machine-learning artifacts are managed using DVC rather than being committed directly to Git.

Important artifacts include:

```text
models/final_best_model.pkl
models/final_prediction_threshold.txt
```

To retrieve DVC-managed artifacts in an authenticated development environment:

```powershell
dvc pull
```

Sensitive Google service-account credentials are stored as CI/CD secrets and must never be committed to the repository.

---

## CI/CD

A push to the `main` branch triggers GitHub Actions.

The CD workflow:

1. Checks out the repository
2. Configures Python
3. Installs DVC Google Drive support
4. Authenticates to the DVC remote
5. Pulls required model artifacts
6. Verifies the production model files
7. Authenticates to Docker Hub
8. Builds the Docker image
9. Pushes the image to Docker Hub

Render then runs the deployed container as the production API service.

---

## Frontend Workflow

Users can provide tumor measurements through three input methods:

1. **Upload CSV**
2. **Load Sample Case**
3. **Manual Entry**

After selecting an input method, the user chooses **Analyze Tumor Risk**.

The frontend sends the 30 diagnostic measurements to the deployed API and displays:

- Classification
- Malignant probability
- Risk level
- Decision threshold
- Explainable AI factors

---

## Project Structure

```text
breast-cancer-mlops/
|
|-- app/
|-- src/
|-- models/
|-- data/
|-- reports/
|-- frontend/
|   |-- index.html
|   |-- style.css
|   |-- script.js
|   |-- sample_case.csv
|   `-- test_patient.csv
|
|-- .github/
|   `-- workflows/
|
|-- Dockerfile
|-- requirements.txt
|-- dvc.yaml
|-- .dvc/
`-- README.md
```

---

## Deployment Status

| Component | Status |
|---|---|
| ML Model | Operational |
| FastAPI Backend | Operational |
| Docker Image | Operational |
| Render Deployment | Operational |
| GitHub Pages Frontend | Operational |
| CSV Input | Tested |
| Sample Case | Tested |
| Manual Entry | Tested |
| SHAP Explainability | Operational |
| CI Pipeline | Passing |
| CD Pipeline | Passing |

---

## Safety and Clinical Disclaimer

This application is an **academic clinical decision-support prototype**.

Predictions and explainability outputs must not be interpreted as medical diagnoses.

The system does not replace:

- qualified healthcare professionals,
- pathology,
- imaging,
- biopsy,
- clinical examination, or
- other validated diagnostic procedures.

Any real-world medical decision must be made by appropriately qualified healthcare professionals.