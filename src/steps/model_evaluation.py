import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from zenml import step
from loguru import logger
import mlflow
from src.models.bike_demand_model import BikeDemandModel


@step
def model_evaluation(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Evaluate the trained model and generate comprehensive evaluation report.

    Args:
        model: Trained model object
        X_test: Test features
        y_test: Test target
        config: Model configuration

    Returns:
        Dictionary containing evaluation metrics and report
    """
    try:
        logger.info("Starting model evaluation")

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred)

        # Generate evaluation report
        evaluation_report = {
            'metrics': metrics,
            'data_info': {
                'test_samples': len(X_test),
                'features': list(X_test.columns)
            },
            'model_info': {
                'type': config.get('model', {}).get('type', 'unknown'),
                'version': config.get('model', {}).get('version', '1.0.0')
            }
        }

        # Generate plots
        plots_dir = Path("reports/figures")
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Residual plot
        create_residual_plot(y_test, y_pred, plots_dir / "residual_plot.png")

        # Prediction vs Actual plot
        create_prediction_plot(y_test, y_pred, plots_dir / "prediction_plot.png")

        # Feature importance plot (if available)
        if hasattr(model, 'feature_importances_'):
            create_feature_importance_plot(
                model.feature_importances_,
                X_test.columns,
                plots_dir / "feature_importance.png"
            )

        # Save evaluation report
        report_path = Path("reports/evaluation_report.json")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, 'w') as f:
            json.dump(evaluation_report, f, indent=2, default=str)

        # Model explainability analysis
        explainability_report = {}
        if isinstance(model, BikeDemandModel) and model.explainer is not None:
            logger.info("Generating model explainability report")

            # Generate comprehensive explainability report
            reports_dir = Path("reports/explainability")
            explainability_report = model.explainer.generate_explainability_report(
                model.model, X_train, X_test, y_test, reports_dir
            )

            # Add explainability summary to evaluation report
            evaluation_report['explainability'] = {
                'global_feature_importance': list(explainability_report.get('global_explanations', {}).get('shap_feature_importance', {}).get('feature_importance', {}).keys())[:5],
                'top_features': explainability_report.get('feature_insights', {}).get('top_features', []),
                'report_path': str(reports_dir / "explainability_report.json")
            }

            logger.info(f"Explainability report generated and saved to {reports_dir}")

        # Model performance monitoring
        if isinstance(model, BikeDemandModel):
            performance_results = model.monitor_performance(X_test, y_test)
            evaluation_report['performance_monitoring'] = performance_results

            # Log evaluation results to MLflow
            eval_run_id = model.log_to_mlflow(
                metrics,
                run_name="model_evaluation",
                experiment_name="bike_demand_model_evaluation"
            )
            if eval_run_id:
                logger.info(f"Evaluation results logged to MLflow with run ID: {eval_run_id}")

            logger.info(f"Performance monitoring completed. Drop detected: {performance_results.get('performance_drop', False)}")

        logger.info(f"Model evaluation completed. MAE: {metrics['mae']:.2f}, R2: {metrics['r2']:.3f}")
        logger.info(f"Evaluation report saved to {report_path}")

        return evaluation_report

    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise


def calculate_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive evaluation metrics.

    Args:
        y_true: True target values
        y_pred: Predicted target values

    Returns:
        Dictionary of evaluation metrics
    """
    try:
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
        mpe = np.mean((y_true - y_pred) / y_true) * 100  # Mean Percentage Error

        # Calculate accuracy within different tolerance levels
        tolerance_levels = [0.1, 0.2, 0.3]  # 10%, 20%, 30% tolerance
        accuracy_metrics = {}
        for tol in tolerance_levels:
            within_tolerance = np.abs((y_true - y_pred) / y_true) <= tol
            accuracy_metrics[f'accuracy_{int(tol*100)}pct'] = within_tolerance.mean()

        metrics = {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mape': mape,
            'mpe': mpe,
            **accuracy_metrics
        }

        return metrics

    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        raise


def create_residual_plot(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    """
    Create and save residual plot.

    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    try:
        residuals = y_true - y_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, residuals, alpha=0.5)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Residual plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating residual plot: {str(e)}")
        raise


def create_prediction_plot(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
    """
    Create and save prediction vs actual plot.

    Args:
        y_true: True values
        y_pred: Predicted values
        save_path: Path to save the plot
    """
    try:
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5, color='blue', label='Predictions')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
                'r--', linewidth=2, label='Perfect Prediction')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Prediction plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating prediction plot: {str(e)}")
        raise


def create_feature_importance_plot(feature_importance: np.ndarray,
                                 feature_names: pd.Index,
                                 save_path: Path) -> None:
    """
    Create and save feature importance plot.

    Args:
        feature_importance: Feature importance scores
        feature_names: Feature names
        save_path: Path to save the plot
    """
    try:
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)
        pos = np.arange(sorted_idx.shape[0]) + .5

        plt.figure(figsize=(10, 6))
        plt.barh(pos, feature_importance[sorted_idx], align='center')
        plt.yticks(pos, feature_names[sorted_idx])
        plt.xlabel('Feature Importance')
        plt.title('Feature Importance Plot')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"Feature importance plot saved to {save_path}")

    except Exception as e:
        logger.error(f"Error creating feature importance plot: {str(e)}")
        raise
