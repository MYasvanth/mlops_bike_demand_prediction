import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
import subprocess
import time
from unittest.mock import patch, MagicMock


class TestEndToEnd:
    """End-to-end tests for the complete MLOps system."""

    @pytest.fixture
    def e2e_config(self):
        """Configuration for end-to-end testing."""
        return {
            'model': {
                'name': 'e2e_test_model',
                'type': 'random_forest',
                'version': '1.0.0'
            },
            'hyperparameters': {
                'n_estimators': 50,
                'max_depth': 10,
                'random_state': 42
            },
            'features': {
                'numerical': ['temp', 'atemp', 'hum', 'windspeed'],
                'categorical': ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday', 'weathersit']
            },
            'target': 'cnt',
            'data': {
                'train_split': 0.7,
                'val_split': 0.15,
                'test_split': 0.15,
                'random_state': 42
            },
            'evaluation': {
                'metrics': ['mae', 'rmse', 'r2'],
                'cv_folds': 5
            },
            'monitoring': {
                'data_quality': {
                    'enable_checks': True,
                    'missing_threshold': 0.05,
                    'duplicate_threshold': 0.01
                },
                'data_drift': {
                    'enable_detection': True,
                    'drift_threshold': 0.1
                },
                'model_performance': {
                    'enable_monitoring': True,
                    'performance_drop_threshold': 0.1
                }
            }
        }

    @pytest.fixture
    def real_dataset(self, tmp_path):
        """Create a realistic dataset similar to bike sharing data."""
        np.random.seed(42)
        n_samples = 1000

        # Create realistic bike sharing features
        data = {
            'temp': np.random.normal(0.5, 0.3, n_samples).clip(0, 1),
            'atemp': np.random.normal(0.5, 0.3, n_samples).clip(0, 1),
            'hum': np.random.normal(0.6, 0.2, n_samples).clip(0, 1),
            'windspeed': np.random.normal(0.2, 0.15, n_samples).clip(0, 1),
            'season': np.random.choice([1, 2, 3, 4], n_samples, p=[0.25, 0.25, 0.25, 0.25]),
            'yr': np.random.choice([0, 1], n_samples, p=[0.5, 0.5]),
            'mnth': np.random.choice(range(1, 13), n_samples),
            'holiday': np.random.choice([0, 1], n_samples, p=[0.95, 0.05]),
            'weekday': np.random.choice(range(7), n_samples),
            'workingday': np.random.choice([0, 1], n_samples, p=[0.3, 0.7]),
            'weathersit': np.random.choice([1, 2, 3], n_samples, p=[0.7, 0.25, 0.05])
        }

        # Generate target with realistic correlations
        base_cnt = 3000
        cnt = (base_cnt +
               data['temp'] * 2000 +           # Warmer weather increases demand
               data['atemp'] * 1000 +          # Feels-like temperature effect
               -data['hum'] * 1500 +           # Humidity decreases demand
               -data['windspeed'] * 1000 +     # Wind decreases demand
               data['season'] * 200 +          # Seasonal effects
               data['yr'] * 500 +              # Year-over-year growth
               -data['holiday'] * 800 +        # Less demand on holidays
               data['workingday'] * 300 +      # More demand on workdays
               -data['weathersit'] * 400)      # Weather situation effects

        # Add noise and ensure positive values
        data['cnt'] = np.maximum(0, cnt + np.random.normal(0, 300, n_samples))

        df = pd.DataFrame(data)

        # Save to temporary file
        data_path = tmp_path / "bike_data.csv"
        df.to_csv(data_path, index=False)

        return str(data_path)

    def test_complete_ml_pipeline(self, e2e_config, real_dataset):
        """Test the complete ML pipeline from data to model."""
        from src.pipelines.training_pipeline import TrainingPipeline

        # Load data
        data = pd.read_csv(real_dataset)

        # Initialize and run pipeline
        pipeline = TrainingPipeline(e2e_config)
        results = pipeline.run(data)

        # Assertions
        assert 'model' in results
        assert 'metrics' in results
        assert 'feature_names' in results
        assert results['model'].is_trained
        assert len(results['feature_names']) > 0
        assert 'mae' in results['metrics']
        assert 'rmse' in results['metrics']
        assert 'r2' in results['metrics']

        # Model should have reasonable performance
        assert results['metrics']['mae'] > 0
        assert results['metrics']['r2'] > 0  # Should explain some variance

    def test_model_persistence_and_loading(self, e2e_config, real_dataset, tmp_path):
        """Test model training, saving, loading, and inference."""
        from src.models.bike_demand_model import BikeDemandModel

        # Train model
        model = BikeDemandModel(e2e_config)
        data = pd.read_csv(real_dataset)
        processed_data = model.preprocess_data(data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        model.train(X_train, y_train)

        # Save model
        model_path = tmp_path / "saved_model.joblib"
        model.save_model(str(model_path))

        # Load model in new instance
        new_model = BikeDemandModel(e2e_config)
        new_model.load_model(str(model_path))

        # Test inference
        predictions = new_model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(pred >= 0 for pred in predictions)  # Bike counts should be non-negative

        # Evaluate loaded model
        eval_metrics = new_model.evaluate(X_test, y_test)
        assert 'mae' in eval_metrics
        assert eval_metrics['mae'] > 0

    def test_data_quality_monitoring_e2e(self, e2e_config, real_dataset):
        """End-to-end test of data quality monitoring."""
        from src.features.data_processor import DataProcessor
        from src.monitoring.alerts import AlertSystem

        data = pd.read_csv(real_dataset)
        processor = DataProcessor(e2e_config)
        alert_system = AlertSystem()

        # Test with clean data
        processor.validate_data(data)
        processed_data = processor.preprocess_features(data, fit=True)

        # Introduce data quality issues
        dirty_data = data.copy()

        # Add missing values
        dirty_data.loc[0:10, 'temp'] = np.nan

        # Add duplicates
        duplicate_rows = dirty_data.iloc[0:5].copy()
        dirty_data = pd.concat([dirty_data, duplicate_rows], ignore_index=True)

        # Test quality monitoring (should detect issues)
        alerts_triggered = []

        # Mock alert system to capture alerts
        original_send_alert = alert_system.send_alert
        def mock_send_alert(subject, message, alert_type="info"):
            alerts_triggered.append((subject, message, alert_type))
            return original_send_alert(subject, message, alert_type)

        alert_system.send_alert = mock_send_alert

        # This would trigger alerts in a real pipeline
        # For testing, we manually check the conditions
        missing_pct = dirty_data['temp'].isnull().sum() / len(dirty_data)
        duplicate_pct = dirty_data.duplicated().sum() / len(dirty_data)

        if missing_pct > e2e_config['monitoring']['data_quality']['missing_threshold']:
            alert_system.alert_data_quality_issue(
                f"High missing values: {missing_pct:.1%} in temp column"
            )

        if duplicate_pct > e2e_config['monitoring']['data_quality']['duplicate_threshold']:
            alert_system.alert_data_quality_issue(
                f"High duplicates: {duplicate_pct:.1%} of dataset"
            )

        # Verify alerts were triggered
        assert len(alerts_triggered) >= 1

    def test_model_serving_simulation(self, e2e_config, real_dataset, tmp_path):
        """Test model serving simulation."""
        from src.models.bike_demand_model import BikeDemandModel

        # Train and save model
        model = BikeDemandModel(e2e_config)
        data = pd.read_csv(real_dataset)
        processed_data = model.preprocess_data(data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        model.train(X_train, y_train)

        model_path = tmp_path / "serving_model.joblib"
        model.save_model(str(model_path))

        # Simulate serving by loading model and making predictions
        serving_model = BikeDemandModel(e2e_config)
        serving_model.load_model(str(model_path))

        # Test batch predictions
        batch_predictions = serving_model.predict(X_test.head(10))
        assert len(batch_predictions) == 10

        # Test single prediction
        single_prediction = serving_model.predict(X_test.head(1))
        assert len(single_prediction) == 1

        # Test with edge cases
        edge_case_data = X_test.head(1).copy()
        edge_case_data.iloc[0] = [0, 0, 0, 0] + [1] * (len(edge_case_data.columns) - 4)  # Min values
        edge_predictions = serving_model.predict(edge_case_data)
        assert len(edge_predictions) == 1
        assert edge_predictions[0] >= 0  # Should handle edge cases gracefully

    @pytest.mark.slow
    def test_pipeline_with_mlflow_tracking(self, e2e_config, real_dataset):
        """Test pipeline with MLflow experiment tracking."""
        from src.pipelines.training_pipeline import TrainingPipeline

        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.sklearn.log_model') as mock_log_model, \
             patch('mlflow.log_artifact') as mock_log_artifact:

            # Mock MLflow run
            mock_run = MagicMock()
            mock_start_run.return_value.__enter__.return_value = mock_run

            # Run pipeline
            data = pd.read_csv(real_dataset)
            pipeline = TrainingPipeline(e2e_config)
            results = pipeline.run(data)

            # Verify MLflow integration
            assert mock_start_run.called
            assert mock_log_params.called
            assert mock_log_metrics.called
            assert mock_log_model.called

    def test_performance_regression_detection(self, e2e_config, real_dataset):
        """Test detection of performance regression."""
        from src.models.bike_demand_model import BikeDemandModel
        from src.monitoring.alerts import AlertSystem

        # Train baseline model
        model = BikeDemandModel(e2e_config)
        data = pd.read_csv(real_dataset)
        processed_data = model.preprocess_data(data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        model.train(X_train, y_train)

        # Get baseline performance
        baseline_metrics = model.evaluate(X_test, y_test)

        # Simulate performance degradation by training on noisy data
        noisy_data = processed_data.copy()
        noisy_data['cnt'] = noisy_data['cnt'] + np.random.normal(0, 1000, len(noisy_data))

        # Retrain on noisy data
        degraded_model = BikeDemandModel(e2e_config)
        X_train_noisy = noisy_data.drop(columns=[e2e_config['target']])
        y_train_noisy = noisy_data[e2e_config['target']]
        degraded_model.train(X_train_noisy, y_train_noisy)

        # Evaluate degraded model
        degraded_metrics = degraded_model.evaluate(X_test, y_test)

        # Check for performance drop
        alert_system = AlertSystem()
        alerts_triggered = []

        original_send_alert = alert_system.send_alert
        def mock_send_alert(subject, message, alert_type="info"):
            alerts_triggered.append((subject, message, alert_type))

        alert_system.send_alert = mock_send_alert

        # Check each metric for degradation
        for metric_name in ['mae', 'rmse']:
            if degraded_metrics[metric_name] > baseline_metrics[metric_name] * 1.2:  # 20% degradation
                alert_system.alert_model_performance_drop(
                    metric_name,
                    degraded_metrics[metric_name],
                    baseline_metrics[metric_name]
                )

        # Should have detected performance drop
        assert len(alerts_triggered) > 0

    def test_scalability_with_larger_dataset(self, e2e_config):
        """Test pipeline scalability with larger dataset."""
        # Generate larger dataset
        np.random.seed(42)
        n_samples = 5000  # Larger dataset

        data = {
            'temp': np.random.normal(0.5, 0.3, n_samples).clip(0, 1),
            'atemp': np.random.normal(0.5, 0.3, n_samples).clip(0, 1),
            'hum': np.random.normal(0.6, 0.2, n_samples).clip(0, 1),
            'windspeed': np.random.normal(0.2, 0.15, n_samples).clip(0, 1),
            'season': np.random.choice([1, 2, 3, 4], n_samples),
            'yr': np.random.choice([0, 1], n_samples),
            'mnth': np.random.choice(range(1, 13), n_samples),
            'holiday': np.random.choice([0, 1], n_samples),
            'weekday': np.random.choice(range(7), n_samples),
            'workingday': np.random.choice([0, 1], n_samples),
            'weathersit': np.random.choice([1, 2, 3], n_samples),
            'cnt': np.random.normal(4000, 1500, n_samples).clip(0, None)
        }

        large_df = pd.DataFrame(data)

        from src.pipelines.training_pipeline import TrainingPipeline

        # Time the pipeline execution
        import time
        start_time = time.time()

        pipeline = TrainingPipeline(e2e_config)
        results = pipeline.run(large_df)

        end_time = time.time()
        execution_time = end_time - start_time

        # Pipeline should complete within reasonable time (adjust threshold as needed)
        assert execution_time < 60  # Less than 1 minute for 5000 samples
        assert results['model'].is_trained

    def test_error_recovery_and_logging(self, e2e_config, caplog):
        """Test error recovery and comprehensive logging."""
        from src.pipelines.training_pipeline import TrainingPipeline

        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid_column': [1, 2, 3]})

        pipeline = TrainingPipeline(e2e_config)

        # Should handle errors gracefully
        with pytest.raises((ValueError, KeyError)):
            pipeline.run(invalid_data)

        # Check that errors were logged
        assert any("ERROR" in record.levelname for record in caplog.records)

    def test_configuration_override(self, e2e_config, real_dataset):
        """Test configuration override capabilities."""
        from src.pipelines.training_pipeline import TrainingPipeline

        # Modify config for testing
        test_config = e2e_config.copy()
        test_config['hyperparameters']['n_estimators'] = 5  # Very small for speed

        data = pd.read_csv(real_dataset)
        pipeline = TrainingPipeline(test_config)
        results = pipeline.run(data)

        # Verify config was used
        assert results['model'].model.n_estimators == 5
        assert results['model'].is_trained
