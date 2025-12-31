#!/usr/bin/env python3
"""
Script to evaluate a saved bike demand model on test data.
This demonstrates comprehensive model evaluation including metrics, plots, and reports.
"""

import sys
from pathlib import Path
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.bike_demand_model import BikeDemandModel


def main():
    """Main function to evaluate saved model."""
    try:
        logger.info("Starting comprehensive model evaluation")

        # Load configuration
        config_path = Path("../configs/model_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize model
        model = BikeDemandModel(config)

        # Load trained model
        model_path = Path("../models/bike_demand_model.joblib")
        if not model_path.exists():
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please train the model first.")

        model.load_model(str(model_path))
        logger.info("Model loaded successfully")

        # Load and preprocess data
        data_path = Path("../data/raw/day.csv")
        df_raw = model.load_data(str(data_path))
        df_processed = model.preprocess_data(df_raw)

        # Split data (same split as training)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(df_processed)

        # Evaluate on all sets
        logger.info("Evaluating model on different datasets...")

        train_metrics = model.evaluate(X_train, y_train, prefix="train")
        val_metrics = model.evaluate(X_val, y_val, prefix="val")
        test_metrics = model.evaluate(X_test, y_test, prefix="test")

        logger.info("Evaluation Metrics:")
        logger.info(f"Train - MAE: {train_metrics['train_mae']:.2f}, R2: {train_metrics['train_r2']:.3f}")
        logger.info(f"Val   - MAE: {val_metrics['val_mae']:.2f}, R2: {val_metrics['val_r2']:.3f}")
        logger.info(f"Test  - MAE: {test_metrics['test_mae']:.2f}, R2: {test_metrics['test_r2']:.3f}")

        # Generate evaluation plots
        reports_dir = Path("../reports/figures")
        reports_dir.mkdir(parents=True, exist_ok=True)

        # Predictions vs Actual plot for test set
        create_predictions_plot(y_test, model.predict(X_test), reports_dir / "test_predictions.png")

        # Residual plot
        create_residual_plot(y_test, model.predict(X_test), reports_dir / "test_residuals.png")

        # Feature importance plot
        feature_importance = model.get_feature_importance()
        create_feature_importance_plot(feature_importance, reports_dir / "feature_importance.png")

        # Save evaluation report
        evaluation_report = {
            "metrics": {
                "train": train_metrics,
                "validation": val_metrics,
                "test": test_metrics
            },
            "data_info": {
                "total_samples": len(df_processed),
                "train_samples": len(X_train),
                "val_samples": len(X_val),
                "test_samples": len(X_test),
                "features": list(X_train.columns)
            },
            "model_info": {
                "type": "RandomForestRegressor",
                "hyperparameters": config.get('hyperparameters', {})
            }
        }

        report_path = Path("../reports/evaluation_report.json")
        import json
        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)

        logger.info(f"Evaluation report saved to {report_path}")
        logger.info("Comprehensive evaluation completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        return 1


def create_predictions_plot(y_true, y_pred, save_path):
    """Create predictions vs actual plot."""
    plt.figure(figsize=(10, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, color='blue')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2)
    plt.xlabel('Actual Bike Demand')
    plt.ylabel('Predicted Bike Demand')
    plt.title('Predicted vs Actual Bike Demand (Test Set)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Predictions plot saved to {save_path}")


def create_residual_plot(y_true, y_pred, save_path):
    """Create residual plot."""
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot (Test Set)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Residual plot saved to {save_path}")


def create_feature_importance_plot(feature_importance, save_path):
    """Create feature importance plot."""
    features = list(feature_importance.keys())
    importance = list(feature_importance.values())

    # Sort by importance
    sorted_idx = sorted(range(len(importance)), key=lambda i: importance[i], reverse=True)
    features_sorted = [features[i] for i in sorted_idx]
    importance_sorted = [importance[i] for i in sorted_idx]

    plt.figure(figsize=(12, 6))
    plt.barh(range(len(features_sorted)), importance_sorted, align='center')
    plt.yticks(range(len(features_sorted)), features_sorted)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance (Random Forest)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"Feature importance plot saved to {save_path}")


if __name__ == "__main__":
    sys.exit(main())
