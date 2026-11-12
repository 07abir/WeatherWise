from datetime import date
from pathlib import Path
import warnings

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# These are the coordinates represented in the training data.
# The saved scaler confirms that latitude and longitude had zero variance.
KOLKATA_LATITUDE = 22.5726
KOLKATA_LONGITUDE = 88.3639
SUPPORTED_LOCATION = "Kolkata"

def load_artifact(filename):
    path = MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing model artifact: {path}")
    # Only load pickle/joblib files that you trust and supplied yourself.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return joblib.load(path)

model = load_artifact("weather_model.pkl")
feature_columns = load_artifact("feature_columns.pkl")
imputer = load_artifact("imputer.pkl")
scaler = load_artifact("scaler.pkl")
label_encoder = load_artifact("label_encoder.pkl")

EXPECTED_FEATURES = ["month", "day_of_year", "latitude", "longitude"]
if list(feature_columns) != EXPECTED_FEATURES:
    raise RuntimeError(
        f"Feature mismatch. Expected {EXPECTED_FEATURES}, got {feature_columns}"
    )

def decode_class(value):
    if value in label_encoder.classes_:
        return str(value)
    return str(label_encoder.inverse_transform([int(value)])[0])


app = FastAPI(
    title="WeatherWise API",
    description=(
        "Kolkata-specific historical-pattern weather classification. "
        "This is not a meteorological forecast."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    prediction_date: date = Field(..., description="Date to classify using historical seasonal patterns")
    location: str = Field(default="Kolkata", description="Only Kolkata is supported by this trained model")

@app.get("/")
def home():
    return {
        "message": "WeatherWise API is running",
        "docs": "/docs",
        "supported_location": SUPPORTED_LOCATION,
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": type(model).__name__,
        "features": list(feature_columns),
        "classes": [str(c) for c in label_encoder.classes_],
        "supported_location": SUPPORTED_LOCATION,
    }

@app.post("/predict")
def predict_weather(payload: PredictionRequest):
    if payload.location.strip().lower() not in {"kolkata", "kolkata, india"}:
        raise HTTPException(
            status_code=400,
            detail=(
                "This model was trained using Kolkata data only. "
                "Please select Kolkata; retrain with multiple cities before supporting other locations."
            ),
        )

    dt = payload.prediction_date
    row = pd.DataFrame([{
        "month": dt.month,
        "day_of_year": dt.timetuple().tm_yday,
        "latitude": KOLKATA_LATITUDE,
        "longitude": KOLKATA_LONGITUDE,
    }], columns=feature_columns)

    try:
        # Apply exactly the same preprocessing objects used during training.
        transformed = imputer.transform(row)
        transformed = scaler.transform(transformed)

        predicted_class = model.predict(transformed)[0]
        predicted_label = decode_class(predicted_class)

        scores = {}
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(transformed)[0]
            for model_class, score in zip(model.classes_, probabilities):
                label = decode_class(model_class)
                scores[label] = round(float(score), 4)

        return {
            "location": SUPPORTED_LOCATION,
            "date": dt.isoformat(),
            "prediction": predicted_label,
            "class_scores": scores,
            "model": type(model).__name__,
            "interpretation": (
                "This result reflects patterns learned from historical observations "
                "and calendar date. It is not a real-time or meteorological forecast."
            ),
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")
