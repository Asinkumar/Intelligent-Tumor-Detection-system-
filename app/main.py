from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.predict import predict_from_features


app = FastAPI(
    title="Intelligent Tumor Decision Support System",
    description=(
        "AI-powered Clinical Decision Support System "
        "for Breast Cancer Risk Prediction"
    ),
    version="1.0.0",
)


# -------------------------------------------------------
# CORS Configuration
# -------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------
# Request Schema
# -------------------------------------------------------

class TumorFeatures(BaseModel):
    features: list[float]


# -------------------------------------------------------
# Home API
# -------------------------------------------------------

@app.get("/")
def home():

    return {
        "project": "Intelligent Tumor Decision Support System",
        "description": (
            "AI-powered Clinical Decision Support System "
            "for Breast Cancer Risk Prediction"
        ),
        "version": "1.0.0",
        "status": "Running Successfully",
    }


# -------------------------------------------------------
# Health Check API
# -------------------------------------------------------

@app.get("/health")
def health():

    return {
        "status": "Healthy",
        "message": "API is running successfully",
    }


# -------------------------------------------------------
# Prediction API
# -------------------------------------------------------

@app.post("/predict")
def predict(data: TumorFeatures):

    if len(data.features) != 30:

        return {
            "status": "Error",
            "message": "Exactly 30 input features are required.",
        }

    result = predict_from_features(data.features)

    return result