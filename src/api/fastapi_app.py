"""
FastAPI application for serving the bike demand prediction model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np
from typing import Dict, List
import joblib
from loguru import logger

from models.bike_demand_model import BikeDemandModel
from features.data_processor import DataProcessor
from utils import load_config

# Initialize FastAPI app
app = FastAPI(
    title="Bike Demand Prediction API",
    description="MLOps API for predicting bike rental demand",
    version="1.0.0"
)

# Global variables for model and processor
model = None
processor = None
config = None

class PredictionInput(BaseModel):
    """Input schema for prediction requests."""
    season: int = Field(..., ge=1, le=4, description="Season (1-4)")
    yr: int = Field(..., ge=0, le=1, description="Year (0: 2011, 1: 2012)")
    mnth: int = Field(..., ge=1, le=12, description="Month (1-12)")
    holiday: int = Field(..., ge=0, le=1, description="Holiday (0: No, 1: Yes)")
    weekday: int = Field(..., ge=0, le=6, description="Weekday (0-6)")
    workingday: int = Field(..., ge=0, le=1, description="Working day (0: No, 1: Yes)")
    weathersit: int = Field(..., ge=1, le=4, description="Weather situation (1-4)")
    temp: float = Field(..., ge=0.0, le=1.0, description="Normalized temperature")
    atemp: float = Field(..., ge=0.0, le=1.0, description="Normalized feeling temperature")
    hum: float = Field(..., ge=0.0, le=1.0, description="Normalized humidity")
    windspeed: float = Field(..., ge=0.0, le=1.0, description="Normalized wind speed")

class PredictionOutput(BaseModel):
    """Output schema for prediction responses."""
    prediction: float = Field(..., description="Predicted bike demand")
    confidence_interval: Dict[str, float] = Field(..., description="Prediction confidence interval")
    feature_importance: Dict[str, float] = Field(..., description="Top feature contributions")

class HealthResponse(BaseModel):
    """Health check response schema."""
    status: str
    model_loaded: bool
    version: str

@app.on_event("startup")
async def startup_event():
    """Load model and processor on startup."""
    global model, processor, config
    
    try:
        logger.info("Loading model and configuration...")
        
        # Load configuration
        config = load_config("configs/model_config.yaml")
        
        # Initialize and load model
        model = BikeDemandModel(config)
        model.load_model("models/bike_demand_model.joblib")
        
        # Initialize and load processor
        processor = DataProcessor(config)
        processor.load_transformers("models/transformers")
        
        logger.success("Model and processor loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        model = None
        processor = None

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if model is not None else "unhealthy",
        model_loaded=model is not None,
        version="1.0.0"
    )

@app.post("/predict", response_model=PredictionOutput)
async def predict(input_data: PredictionInput):
    """Make bike demand prediction."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert input to DataFrame
        input_dict = input_data.dict()
        df = pd.DataFrame([input_dict])
        
        # Preprocess the data
        df_processed = processor.preprocess_features(df, fit=False)
        df_processed = processor.create_features(df_processed, fit=False)
        
        # Make prediction
        prediction = model.predict(df_processed)[0]
        
        # Calculate confidence interval (simple approach using model uncertainty)
        # In production, you might use more sophisticated uncertainty quantification
        std_error = prediction * 0.1  # Assume 10% standard error
        confidence_interval = {
            "lower": max(0, prediction - 1.96 * std_error),
            "upper": prediction + 1.96 * std_error
        }
        
        # Get feature importance for this prediction
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            # Get top 5 features
            top_features = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:5])
        else:
            top_features = {}
        
        return PredictionOutput(
            prediction=float(prediction),
            confidence_interval=confidence_interval,
            feature_importance=top_features
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict_batch")
async def predict_batch(input_data: List[PredictionInput]):
    """Make batch predictions."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert inputs to DataFrame
        input_dicts = [item.dict() for item in input_data]
        df = pd.DataFrame(input_dicts)
        
        # Preprocess the data
        df_processed = processor.preprocess_features(df, fit=False)
        df_processed = processor.create_features(df_processed, fit=False)
        
        # Make predictions
        predictions = model.predict(df_processed)
        
        return {
            "predictions": predictions.tolist(),
            "count": len(predictions)
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get model information."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        info = {
            "model_type": "RandomForestRegressor",
            "features": model.feature_names if hasattr(model, 'feature_names') else [],
            "target": config.get('target', 'cnt'),
            "version": config.get('model', {}).get('version', '1.0.0')
        }
        
        if hasattr(model.model, 'n_estimators'):
            info["n_estimators"] = model.model.n_estimators
        if hasattr(model.model, 'max_depth'):
            info["max_depth"] = model.model.max_depth
            
        return info
        
    except Exception as e:
        logger.error(f"Model info error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)