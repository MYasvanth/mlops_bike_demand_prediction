#!/usr/bin/env python3
"""
Script to load a trained bike demand model and make predictions.
This demonstrates how to load and use a pre-trained model for inference.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.bike_demand_model import BikeDemandModel


def main():
    """Main function to load model and make predictions."""
    try:
        logger.info("Starting bike demand model loading and prediction")

        # Load configuration
        config_path = Path("../configs/model_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize model (without training)
        model = BikeDemandModel(config)

        # Load trained model
        model_path = Path("../models/bike_demand_model.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")

        model.load_model(str(model_path))
        logger.info("Model loaded successfully")

        # Load some sample data for prediction (using test data as example)
        data_path = Path("../data/raw/day.csv")
        df_raw = model.load_data(str(data_path))
        df_processed = model.preprocess_data(df_raw)

        # Get a few samples for prediction
        sample_data = df_processed.head(5)  # First 5 rows
        X_sample = sample_data[model.feature_names]
        y_actual = sample_data[model.target_name]

        # Make predictions
        predictions = model.predict(X_sample)

        # Display results
        logger.info("Sample Predictions:")
        for i, (actual, pred) in enumerate(zip(y_actual, predictions)):
            logger.info(f"Sample {i+1}: Actual={actual:.0f}, Predicted={pred:.2f}, Error={abs(actual-pred):.2f}")

        # Calculate prediction metrics on sample
        from sklearn.metrics import mean_absolute_error, r2_score
        mae = mean_absolute_error(y_actual, predictions)
        r2 = r2_score(y_actual, predictions)

        logger.info(f"Sample prediction metrics - MAE: {mae:.2f}, R2: {r2:.3f}")

        # Example of predicting on new data (simulated)
        logger.info("Example: Predicting on new hypothetical data")
        # Create a hypothetical sample
        hypothetical_data = pd.DataFrame({
            'temp': [0.5],      # Normalized temperature
            'atemp': [0.6],     # Normalized feeling temperature
            'hum': [0.7],       # Normalized humidity
            'windspeed': [0.3], # Normalized windspeed
            'season': [2],      # Season (1-4)
            'yr': [1],          # Year (0=2011, 1=2012)
            'mnth': [7],        # Month (1-12)
            'holiday': [0],     # Is holiday
            'weekday': [1],     # Day of week
            'workingday': [1],  # Is working day
            'weathersit': [1]   # Weather situation
        })

        hypothetical_prediction = model.predict(hypothetical_data)
        logger.info(f"Hypothetical prediction: {hypothetical_prediction[0]:.2f} bikes")

        logger.info("Prediction completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in loading and prediction: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
