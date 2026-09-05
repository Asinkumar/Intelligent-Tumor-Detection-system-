# Intelligent Tumor Decision Support System

[![CI](https://github.com/Asinkumar/Intelligent-Tumor-Detection-system-/actions/workflows/ci.yml/badge.svg)](https://github.com/Asinkumar/Intelligent-Tumor-Detection-system-/actions/workflows/ci.yml)

[![CD](https://github.com/Asinkumar/Intelligent-Tumor-Detection-system-/actions/workflows/cd.yml/badge.svg)](https://github.com/Asinkumar/Intelligent-Tumor-Detection-system-/actions/workflows/cd.yml)

[![Docker](https://img.shields.io/badge/Docker-Containerized-blue?logo=docker)](https://hub.docker.com/r/ayesin/tumor-decision-support)

[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)](https://tumor-decision-support.onrender.com/docs)

[![DVC](https://img.shields.io/badge/DVC-Data%20Versioning-945DD6?logo=dvc)](https://dvc.org/)

An end-to-end Explainable AI and MLOps-based clinical decision-support prototype for breast cancer risk classification using the Wisconsin Diagnostic Breast Cancer dataset.

The system predicts whether a tumor record is **Malignant** or **Benign**, estimates malignant probability, assigns a risk level, and provides feature-level explanations for the prediction.

> **Academic Prototype:** This system is intended for research and educational purposes only. It is not clinically validated and must not replace professional medical diagnosis, pathology, imaging, biopsy, or medical advice.

---

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
- Continuous-learning / feedback pipeline
- Model retraining workflow
- Data drift monitoring
- Model promotion safety checks

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

## MLOps Architecture

```text
Training Data
    |
    v
Data Validation
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
DVC Versioning
    |
    +--------------------+
    |                    |
    v                    v
GitHub              Google Drive
    |                DVC Remote
    |
    v
GitHub Actions CI
    |
    +----> Install dependencies
    |
    +----> Pull DVC artifacts
    |
    +----> Verify model files
    |
    +----> Run automated tests
    |
    +----> Build Docker image
    |
    v
GitHub Actions CD
    |
    +----> Pull DVC artifacts
    |
    +----> Docker build
    |
    +----> Push to Docker Hub
    |
    v
Render Deployment
    |
    v
Production FastAPI Service

## Continuous Learning and Monitoring

The project also contains a feedback-driven model maintenance workflow.

```text
Prediction
    |
    v
Doctor / Validated Feedback
    |
    v
Feedback Dataset
    |
    v
Monitoring
    |
    +----> Data Drift Detection
    |
    v
Retraining Trigger
    |
    v
Candidate Model
    |
    v
Performance Validation
    |
    +----> Promote if better
    |
    +----> Reject if unsafe / worse

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

## Technology Stack

### Machine Learning

- Python
- Scikit-learn
- Pandas
- NumPy
- SHAP
- MLflow

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
- Pytest

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

Example response:

```json
{
  "status": "Healthy",
  "message": "API is running successfully"
}
```

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

### Clone the repository

```bash
git clone https://github.com/Asinkumar/Intelligent-Tumor-Detection-system-.git
cd Intelligent-Tumor-Detection-system-
```

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

### Pull DVC artifacts

```powershell
dvc pull
```

### Run the API

```powershell
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Run with Docker

Pull the latest production image:

```bash
docker pull ayesin/tumor-decision-support:latest
```

Run the container:

```bash
docker run -p 8000:8000 ayesin/tumor-decision-support:latest
```

Open:

```text
http://localhost:8000/docs
```

---

## DVC Model Artifacts

Large machine-learning artifacts are managed using DVC rather than being committed directly to Git.

Important artifacts include:

```text
models/final_best_model.pkl
models/final_prediction_threshold.txt
```

Main tracked directories include:

```text
data/
models/
reports/
```

To retrieve DVC-managed artifacts in an authenticated development environment:

```powershell
dvc pull
```

Sensitive Google authentication credentials are stored securely for CI/CD and must never be committed to the repository.

---

## CI Pipeline

A push to the `main` branch triggers the CI workflow.

The CI pipeline:

1. Checks out the repository
2. Configures Python
3. Installs project dependencies
4. Installs DVC with Google Drive support
5. Authenticates to the DVC remote
6. Pulls DVC artifacts
7. Verifies production model files
8. Runs Python syntax validation
9. Verifies important dependencies
10. Runs automated tests
11. Builds the Docker image

Current status: **Passing**

---

## CD Pipeline

The CD workflow:

1. Checks out the repository
2. Configures Python
3. Installs DVC Google Drive support
4. Authenticates to Google Drive
5. Pulls required model artifacts
6. Verifies the production model files
7. Authenticates to Docker Hub
8. Sets up Docker Buildx
9. Builds the production image
10. Pushes Docker images to Docker Hub

Published tags:

```text
ayesin/tumor-decision-support:latest
ayesin/tumor-decision-support:<commit-sha>
```

Current status: **Passing**

---

## Automated Testing

The project includes automated tests for the application and MLOps workflow.

Tests cover areas such as:

- API behavior
- continuous-learning workflow
- feedback processing
- retraining logic
- monitoring logic
- model promotion safeguards

The test suite is executed automatically in GitHub Actions before the Docker build is considered successful.

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
|-- tests/
|-- frontend/
|   |-- index.html
|   |-- style.css
|   |-- script.js
|   |-- sample_case.csv
|   `-- test_patient.csv
|
|-- .github/
|   `-- workflows/
|       |-- ci.yml
|       `-- cd.yml
|
|-- Dockerfile
|-- requirements.txt
|-- dvc.yaml
|-- data.dvc
|-- models.dvc
|-- reports.dvc
|-- .dvc/
|-- .dvcignore
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
| DVC Remote | Operational |
| Google Drive Artifact Storage | Operational |
| CSV Input | Tested |
| Sample Case | Tested |
| Manual Entry | Tested |
| SHAP Explainability | Operational |
| Continuous Learning | Implemented |
| Monitoring / Drift Detection | Implemented |
| Automated Tests | Passing |
| CI Pipeline | Passing |
| CD Pipeline | Passing |
| Docker Hub Push | Passing |

---

## Production URLs

### Frontend
https://asinkumar.github.io/Intelligent-Tumor-Detection-system-/

### Backend
https://tumor-decision-support.onrender.com/

### Health Check
https://tumor-decision-support.onrender.com/health

### API Documentation
https://tumor-decision-support.onrender.com/docs

### Docker Hub
https://hub.docker.com/r/ayesin/tumor-decision-support

---

## Safety and Clinical Disclaimer

This application is an **academic clinical decision-support prototype**.

Predictions and explainability outputs must not be interpreted as medical diagnoses.

The system does not replace:

- qualified healthcare professionals
- pathology
- imaging
- biopsy
- clinical examination
- other validated diagnostic procedures

Any real-world medical decision must be made by appropriately qualified healthcare professionals.

---

## Project Status

**End-to-end MLOps workflow operational.**

The project currently supports:

**Data → Model → DVC → CI → Testing → Docker → CD → Docker Hub → Render → Frontend → Monitoring → Retraining Workflow**

