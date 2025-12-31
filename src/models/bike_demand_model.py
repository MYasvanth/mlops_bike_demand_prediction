import logging
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
from loguru import logger
from src.monitoring.alerts import AlertSystem
from src.visualization.model_explainability import ModelExplainer
import json
import os


class BikeDemandModel:
    """Unified model class for bike demand prediction following ML engineering best practices."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the bike demand model.

        Args:
            config: Model configuration dictionary
        """
        self.config = config
        self.model: Optional[RandomForestRegressor] = None
        self.feature_names: List[str] = []
        self.target_name: str = config.get('target', 'cnt')
        self.is_trained: bool = False
        self.explainer: Optional[ModelExplainer] = None

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format=self.config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load data from CSV file.

        Args:
            data_path: Path to the CSV file

        Returns:
            Loaded DataFrame

        Raises:
            FileNotFoundError: If data file doesn't exist
            ValueError: If data is invalid
        """
        try:
            logger.info(f"Loading data from {data_path}")
            df = pd.read_csv(data_path)

            if df.empty:
                raise ValueError("Loaded dataset is empty")

            logger.info(f"Successfully loaded {len(df)} rows with {len(df.columns)} columns")
            return df

        except FileNotFoundError:
            logger.error(f"Data file not found: {data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess the raw data. This method assumes data has already been preprocessed by DataProcessor.
        It performs final validation and sets feature names.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Validated DataFrame
        """
        try:
            logger.info("Starting model data preprocessing validation")

            # Make a copy to avoid modifying original
            df_processed = df.copy()

            # Handle missing values
            df_processed = df_processed.dropna()

            # Validate required columns exist
            required_cols = self.config['features']['numerical'] + self.config['features']['categorical'] + [self.target_name]
            missing_cols = [col for col in required_cols if col not in df_processed.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Set feature names (use all columns except target)
            self.feature_names = [col for col in df_processed.columns if col != self.target_name]

            logger.info(f"Model data preprocessing validation completed. Shape: {df_processed.shape}")
            return df_processed

        except Exception as e:
            logger.error(f"Error in model data preprocessing: {str(e)}")
            raise

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
        """
        Split data into train, validation, and test sets.

        Args:
            df: Preprocessed DataFrame

        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test)
        """
        try:
            logger.info("Splitting data into train/validation/test sets")

            X = df[self.feature_names]
            y = df[self.target_name]

            # First split: train + (val + test)
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y,
                test_size=(self.config['data']['val_split'] + self.config['data']['test_split']),
                random_state=self.config['data']['random_state']
            )

            # Second split: val and test
            val_size = self.config['data']['val_split'] / (self.config['data']['val_split'] + self.config['data']['test_split'])
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp,
                test_size=(1 - val_size),
                random_state=self.config['data']['random_state']
            )

            logger.info(f"Data split completed. Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
            return X_train, X_val, X_test, y_train, y_val, y_test

        except Exception as e:
            logger.error(f"Error splitting data: {str(e)}")
            raise

    def train(self, X_train: pd.DataFrame, y_train: pd.Series, X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, float]:
        """
        Train the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            Dictionary with training metrics
        """
        try:
            logger.info("Starting model training")

            # Initialize model with hyperparameters
            self.model = RandomForestRegressor(**self.config['hyperparameters'])

            # Perform cross-validation before fitting final model (unbiased estimate)
            cv_scores = cross_val_score(
                self.model, X_train, y_train,
                cv=self.config['evaluation']['cv_folds'],
                scoring='neg_mean_absolute_error'
            )
            cv_mae = -cv_scores.mean()

            # Train the final model on full training data
            self.model.fit(X_train, y_train)
            self.is_trained = True

            # Initialize explainer for interpretability
            if self.config.get('explainability', {}).get('enabled', False):
                self.explainer = ModelExplainer(self.config)
                self.explainer.initialize_explainer(self.model, X_train)
                logger.info("Model explainer initialized")

            # Calculate training metrics
            train_metrics = self.evaluate(X_train, y_train, prefix="train")

            # Calculate validation metrics if provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                val_metrics = self.evaluate(X_val, y_val, prefix="val")

            metrics = {**train_metrics, **val_metrics, 'cv_mae': cv_mae}

            logger.info(f"Model training completed. Train MAE: {train_metrics.get('train_mae', 'N/A'):.2f}, CV MAE: {cv_mae:.2f}")
            return metrics

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def evaluate(self, X: pd.DataFrame, y: pd.Series, prefix: str = "") -> Dict[str, float]:
        """
        Evaluate the model on given data.

        Args:
            X: Features
            y: Target
            prefix: Prefix for metric names

        Returns:
            Dictionary with evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation")

        try:
            y_pred = self.model.predict(X)

            mae = mean_absolute_error(y, y_pred)
            mse = mean_squared_error(y, y_pred)
            rmse = np.sqrt(mse)
            r2 = r2_score(y, y_pred)

            metrics = {
                f"{prefix}_mae" if prefix else "mae": mae,
                f"{prefix}_rmse" if prefix else "rmse": rmse,
                f"{prefix}_r2" if prefix else "r2": r2
            }

            return metrics

        except Exception as e:
            logger.error(f"Error evaluating model: {str(e)}")
            raise

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions with the trained model.

        Args:
            X: Features for prediction

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")

        try:
            return self.model.predict(X)
        except Exception as e:
            logger.error(f"Error making predictions: {str(e)}")
            raise

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before getting feature importance")

        try:
            importance_scores = self.model.feature_importances_
            return dict(zip(self.feature_names, importance_scores))
        except Exception as e:
            logger.error(f"Error getting feature importance: {str(e)}")
            raise

    def explain_prediction(self, X_instance: pd.DataFrame) -> Dict[str, Any]:
        """
        Explain a single prediction using SHAP and LIME.

        Args:
            X_instance: Single instance to explain

        Returns:
            Dictionary with SHAP and LIME explanations
        """
        if not self.is_trained or self.explainer is None:
            raise ValueError("Model must be trained and explainer initialized")

        try:
            logger.info("Generating prediction explanations")

            # SHAP explanation
            shap_explanation = self.explainer.explain_prediction_shap(X_instance)

            # LIME explanation
            lime_explanation = self.explainer.explain_prediction_lime(X_instance)

            explanation = {
                'shap': shap_explanation,
                'lime': lime_explanation,
                'feature_names': self.feature_names
            }

            logger.info("Prediction explanations generated successfully")
            return explanation

        except Exception as e:
            logger.error(f"Error generating prediction explanations: {str(e)}")
            raise

    def get_global_explanations(self, X_test: pd.DataFrame) -> Dict[str, Any]:
        """
        Get global model explanations using SHAP.

        Args:
            X_test: Test data for global explanation

        Returns:
            Dictionary with global SHAP explanations
        """
        if not self.is_trained or self.explainer is None:
            raise ValueError("Model must be trained and explainer initialized")

        try:
            logger.info("Generating global model explanations")

            # Global SHAP feature importance
            global_shap = self.explainer.global_feature_importance_shap(X_test)

            explanations = {
                'shap_feature_importance': global_shap,
                'traditional_feature_importance': self.get_feature_importance()
            }

            logger.info("Global explanations generated successfully")
            return explanations

        except Exception as e:
            logger.error(f"Error generating global explanations: {str(e)}")
            raise

    def save_model(self, path: str) -> None:
        """
        Save the trained model to disk.

        Args:
            path: Path to save the model
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before saving")

        try:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.model, path)
            logger.info(f"Model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, path: str) -> None:
        """
        Load a trained model from disk.

        Args:
            path: Path to the saved model
        """
        try:
            self.model = joblib.load(path)
            self.is_trained = True
            logger.info(f"Model loaded from {path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise

    def log_to_mlflow(self, metrics: Dict[str, float], run_name: str = None, experiment_name: str = "bike_demand_prediction") -> str:
        """
        Log model, metrics, and artifacts to MLflow with comprehensive experiment tracking.

        Args:
            metrics: Dictionary of metrics to log
            run_name: Name for the MLflow run
            experiment_name: Name of the MLflow experiment

        Returns:
            Run ID of the logged experiment
        """
        try:
            import mlflow
            import mlflow.sklearn
            from mlflow.models.signature import infer_signature
            import tempfile
            import shutil

            # Set experiment
            mlflow.set_experiment(experiment_name)

            # Create a temporary directory for artifacts
            with tempfile.TemporaryDirectory() as temp_dir:
                with mlflow.start_run(run_name=run_name) as run:
                    run_id = run.info.run_id

                    # Log run metadata
                    mlflow.set_tag("model_type", "RandomForestRegressor")
                    mlflow.set_tag("model_version", self.config.get('model', {}).get('version', '1.0.0'))
                    mlflow.set_tag("data_source", self.config.get('data', {}).get('raw_path', 'unknown'))
                    mlflow.set_tag("training_date", pd.Timestamp.now().isoformat())
                    mlflow.set_tag("feature_count", len(self.feature_names))
                    mlflow.set_tag("target_variable", self.target_name)

                    # Log hyperparameters
                    hyperparameters = self.config.get('hyperparameters', {})
                    for param_name, param_value in hyperparameters.items():
                        mlflow.log_param(param_name, param_value)

                    # Log data parameters
                    data_config = self.config.get('data', {})
                    mlflow.log_param("train_split", data_config.get('train_split', 0.8))
                    mlflow.log_param("val_split", data_config.get('val_split', 0.1))
                    mlflow.log_param("test_split", data_config.get('test_split', 0.1))
                    mlflow.log_param("random_state", data_config.get('random_state', 42))
                    mlflow.log_param("cv_folds", self.config.get('evaluation', {}).get('cv_folds', 5))

                    # Log feature information
                    mlflow.log_param("numerical_features", ','.join(self.config.get('features', {}).get('numerical', [])))
                    mlflow.log_param("categorical_features", ','.join(self.config.get('features', {}).get('categorical', [])))
                    mlflow.log_param("feature_names", ','.join(self.feature_names))

                    # Log metrics
                    for metric_name, metric_value in metrics.items():
                        mlflow.log_metric(metric_name, metric_value)

                    # Log feature importance if available
                    if self.is_trained and hasattr(self.model, 'feature_importances_'):
                        feature_importance = self.get_feature_importance()
                        # Save feature importance as artifact
                        importance_df = pd.DataFrame({
                            'feature': list(feature_importance.keys()),
                            'importance': list(feature_importance.values())
                        }).sort_values('importance', ascending=False)

                        importance_path = f"{temp_dir}/feature_importance.csv"
                        importance_df.to_csv(importance_path, index=False)
                        mlflow.log_artifact(importance_path, "feature_analysis")

                        # Log top 10 features as parameters
                        top_features = importance_df.head(10)['feature'].tolist()
                        mlflow.log_param("top_10_features", ','.join(top_features))

                    # Create sample input for model signature
                    sample_input = pd.DataFrame({
                        feature: [0.0] for feature in self.feature_names
                    })

                    # Infer model signature
                    try:
                        signature = infer_signature(sample_input, self.model.predict(sample_input))
                    except Exception as e:
                        logger.warning(f"Could not infer model signature: {e}")
                        signature = None

                    # Log the model with comprehensive metadata
                    model_info = mlflow.sklearn.log_model(
                        sk_model=self.model,
                        artifact_path="model",
                        signature=signature,
                        registered_model_name=self.config.get('model', {}).get('name', 'BikeDemandModel'),
                        input_example=sample_input,
                        metadata={
                            "model_type": "RandomForestRegressor",
                            "training_framework": "scikit-learn",
                            "feature_engineering": "standard_scaling + label_encoding",
                            "hyperparameter_tuning": "optuna_optimization",
                            "cross_validation": f"{self.config.get('evaluation', {}).get('cv_folds', 5)}-fold",
                            "data_preprocessing": "IQR_outlier_removal + feature_engineering"
                        }
                    )

                    # Log additional artifacts
                    # Save configuration as artifact
                    config_path = f"{temp_dir}/model_config.yaml"
                    import yaml
                    with open(config_path, 'w') as f:
                        yaml.dump(self.config, f, default_flow_style=False)
                    mlflow.log_artifact(config_path, "config")

                    # Log model summary statistics
                    if hasattr(self.model, 'estimators_'):
                        mlflow.log_metric("num_trees", len(self.model.estimators_))
                        mlflow.log_metric("max_tree_depth", max([tree.get_depth() for tree in self.model.estimators_]))

                    logger.info(f"Successfully logged model to MLflow. Run ID: {run_id}, Model URI: {model_info.model_uri}")

                    return run_id

        except Exception as e:
            logger.error(f"MLflow logging failed: {str(e)}")
            # Don't raise exception to avoid breaking the training pipeline
            return None

    def monitor_performance(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """
        Monitor model performance and alert if performance drops.

        Args:
            X_test: Test features
            y_test: Test target

        Returns:
            Dictionary with performance monitoring results
        """
        try:
            logger.info("Starting model performance monitoring")

            # Calculate current performance
            current_metrics = self.evaluate(X_test, y_test)

            # Load baseline metrics if available
            baseline_path = self.config.get('monitoring', {}).get('model_performance', {}).get('baseline_metrics_path', 'models/baseline_metrics.json')
            baseline_metrics = {}

            if os.path.exists(baseline_path):
                with open(baseline_path, 'r') as f:
                    baseline_metrics = json.load(f)
                logger.info(f"Loaded baseline metrics from {baseline_path}")
            else:
                logger.warning(f"Baseline metrics not found at {baseline_path}. Using current metrics as baseline.")
                # Save current metrics as baseline
                os.makedirs(os.path.dirname(baseline_path), exist_ok=True)
                with open(baseline_path, 'w') as f:
                    json.dump(current_metrics, f, indent=2)
                return {"current_metrics": current_metrics, "baseline_metrics": current_metrics, "performance_drop": False}

            # Check for performance drop
            performance_drop_threshold = self.config.get('monitoring', {}).get('model_performance', {}).get('performance_drop_threshold', 0.1)
            performance_drop = False

            alerts = []
            for metric_name, current_value in current_metrics.items():
                baseline_value = baseline_metrics.get(metric_name)
                if baseline_value is not None:
                    # For MAE and RMSE, higher values indicate worse performance
                    if metric_name in ['mae', 'rmse']:
                        if current_value > baseline_value * (1 + performance_drop_threshold):
                            performance_drop = True
                            alerts.append(f"{metric_name.upper()} increased from {baseline_value:.3f} to {current_value:.3f}")
                    # For R2, lower values indicate worse performance
                    elif metric_name == 'r2':
                        if current_value < baseline_value * (1 - performance_drop_threshold):
                            performance_drop = True
                            alerts.append(f"{metric_name.upper()} decreased from {baseline_value:.3f} to {current_value:.3f}")

            # Alert if performance drop detected
            if performance_drop:
                alert_system = AlertSystem(
                    smtp_server=self.config.get('monitoring', {}).get('alerting', {}).get('smtp_server'),
                    smtp_port=self.config.get('monitoring', {}).get('alerting', {}).get('smtp_port', 587),
                    sender_email=self.config.get('monitoring', {}).get('alerting', {}).get('sender_email'),
                    sender_password=self.config.get('monitoring', {}).get('alerting', {}).get('sender_password'),
                    recipient_emails=self.config.get('monitoring', {}).get('alerting', {}).get('recipient_emails', [])
                )
                alert_system.alert_model_performance_drop(
                    "Multiple metrics" if len(alerts) > 1 else alerts[0].split()[0],
                    current_metrics.get('mae', 0),
                    baseline_metrics.get('mae', 0)
                )

            # Log performance monitoring to MLflow
            with mlflow.start_run(run_name="performance_monitoring", nested=True):
                for metric_name, value in current_metrics.items():
                    mlflow.log_metric(f"current_{metric_name}", value)
                for metric_name, value in baseline_metrics.items():
                    mlflow.log_metric(f"baseline_{metric_name}", value)
                mlflow.log_param("performance_drop_detected", performance_drop)
                if alerts:
                    mlflow.log_param("performance_alerts", "; ".join(alerts))

            logger.info(f"Model performance monitoring completed. Performance drop: {performance_drop}")
            return {
                "current_metrics": current_metrics,
                "baseline_metrics": baseline_metrics,
                "performance_drop": performance_drop,
                "alerts": alerts
            }

        except Exception as e:
            logger.error(f"Error in model performance monitoring: {str(e)}")
            raise

    def save_baseline_metrics(self, metrics: Dict[str, float], path: str) -> None:
        """
        Save baseline metrics for performance monitoring.

        Args:
            metrics: Metrics to save as baseline
            path: Path to save baseline metrics
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, 'w') as f:
                json.dump(metrics, f, indent=2)
            logger.info(f"Baseline metrics saved to {path}")
        except Exception as e:
            logger.error(f"Error saving baseline metrics: {str(e)}")
            raise
