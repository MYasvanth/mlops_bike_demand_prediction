from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import logging
from typing import List, Dict, Any
import uvicorn
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Bike Demand Prediction API", version="1.0.0")

# Load model and transformers
MODEL_PATH = Path("models/bike_demand_model.joblib")
TRANSFORMERS_PATH = Path("models/transformers")

try:
    model = joblib.load(MODEL_PATH)
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None

# Load transformers
transformers = {}
try:
    for file in TRANSFORMERS_PATH.glob("*.joblib"):
        name = file.stem
        transformers[name] = joblib.load(file)
    logger.info(f"Loaded {len(transformers)} transformers")
except Exception as e:
    logger.error(f"Failed to load transformers: {e}")

class PredictionRequest(BaseModel):
    temp: float
    atemp: float
    hum: float
    windspeed: float
    season: int
    yr: int
    mnth: int
    holiday: int
    weekday: int
    workingday: int
    weathersit: int

class PredictionResponse(BaseModel):
    prediction: float
    features_used: List[str]

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Predict bike demand"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        # Convert request to DataFrame
        data = pd.DataFrame([request.dict()])

        # Apply transformations
        for name, transformer in transformers.items():
            if 'scaler' in name:
                col = name.replace('scaler_', '')
                if col in data.columns:
                    data[col] = transformer.transform(data[[col]])
            elif 'encoder' in name:
                col = name.replace('encoder_', '')
                if col in data.columns:
                    data[col] = transformer.transform(data[[col]])

        # Make prediction
        prediction = model.predict(data)[0]

        return PredictionResponse(
            prediction=float(prediction),
            features_used=list(data.columns)
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")

@app.get("/model-info")
async def model_info():
    """Get model information"""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "model_type": str(type(model).__name__),
        "transformers_loaded": len(transformers),
        "transformer_names": list(transformers.keys())
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
