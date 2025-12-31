#!/usr/bin/env python3
"""
Script to train and evaluate the bike demand model using the unified BikeDemandModel class.
This script demonstrates how to load, train, and evaluate the model outside of the ZenML pipeline.
"""

import sys
from pathlib import Path
import yaml
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.bike_demand_model import BikeDemandModel


def main():
    """Main function to train and evaluate the model."""
    try:
        logger.info("Starting bike demand model training and evaluation")

        # Load configuration
        config_path = Path("../configs/model_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize model
        model = BikeDemandModel(config)

        # Load and preprocess data
        data_path = Path("../data/raw/day.csv")
        df_raw = model.load_data(str(data_path))
        df_processed = model.preprocess_data(df_raw)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(df_processed)

        # Train model
        logger.info("Training the model...")
        train_metrics = model.train(X_train, y_train, X_val, y_val)
        logger.info(f"Training completed. Metrics: {train_metrics}")

        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = model.evaluate(X_test, y_test)
        logger.info(f"Test metrics: {test_metrics}")

        # Get feature importance
        feature_importance = model.get_feature_importance()
        logger.info(f"Top 5 important features: {dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)[:5])}")

        # Log to MLflow
        logger.info("Logging to MLflow...")
        all_metrics = {**train_metrics, **test_metrics}
        model.log_to_mlflow(all_metrics, run_name="bike_demand_manual_training")

        # Save model
        model_path = Path("../models/bike_demand_model.joblib")
        model.save_model(str(model_path))
        logger.info(f"Model saved to {model_path}")

        logger.info("Training and evaluation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in training and evaluation: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
