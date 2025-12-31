import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock
from src.features.data_processor import DataProcessor
from src.models.bike_demand_model import BikeDemandModel
from src.pipelines.training_pipeline import TrainingPipeline
from src.monitoring.alerts import AlertSystem


class TestIntegration:
    """Integration tests for the complete MLOps pipeline."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for integration testing."""
        return {
            'model': {
                'name': 'integration_test_model',
                'type': 'random_forest',
                'version': '1.0.0'
            },
            'hyperparameters': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            },
            'features': {
                'numerical': ['temp', 'hum', 'windspeed'],
                'categorical': ['season', 'weathersit', 'weekday']
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
                'cv_folds': 3
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
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }

    @pytest.fixture
    def sample_data(self):
        """Generate realistic bike demand dataset for testing."""
        np.random.seed(42)
        n_samples = 500

        # Generate realistic features
        data = {
            'temp': np.random.normal(0.5, 0.2, n_samples).clip(0, 1),  # Normalized temperature
            'hum': np.random.normal(0.6, 0.15, n_samples).clip(0, 1),  # Normalized humidity
            'windspeed': np.random.normal(0.2, 0.1, n_samples).clip(0, 1),  # Normalized windspeed
            'season': np.random.choice([1, 2, 3, 4], n_samples, p=[0.25, 0.25, 0.25, 0.25]),
            'weathersit': np.random.choice([1, 2, 3], n_samples, p=[0.6, 0.3, 0.1]),
            'weekday': np.random.choice(range(7), n_samples),
            'cnt': np.random.normal(4500, 2000, n_samples).clip(0, None)  # Bike count
        }

        # Add some realistic correlations
        data['cnt'] = (data['cnt'] +
                      data['temp'] * 1000 +  # Higher demand in warmer weather
                      -data['hum'] * 500 +   # Lower demand in humid weather
                      -data['windspeed'] * 300).clip(0, None)

        return pd.DataFrame(data)

    def test_data_processor_model_integration(self, sample_config, sample_data):
        """Test integration between DataProcessor and BikeDemandModel."""
        # Initialize components
        processor = DataProcessor(sample_config)
        model = BikeDemandModel(sample_config)

        # Process data
        processor.validate_data(sample_data)
        processed_data = processor.preprocess_features(sample_data, fit=True)

        # Train model
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        metrics = model.train(X_train, y_train, X_val, y_val)

        # Evaluate model
        eval_metrics = model.evaluate(X_test, y_test)

        # Assertions
        assert processor.is_fitted
        assert model.is_trained
        assert isinstance(metrics, dict)
        assert isinstance(eval_metrics, dict)
        assert 'mae' in eval_metrics
        assert eval_metrics['mae'] > 0  # Should have some error

    def test_full_pipeline_execution(self, sample_config, sample_data, tmp_path):
        """Test complete pipeline execution."""
        # Create temporary directories
        models_dir = tmp_path / "models"
        transformers_dir = tmp_path / "transformers"
        models_dir.mkdir()
        transformers_dir.mkdir()

        # Initialize pipeline
        pipeline = TrainingPipeline(sample_config)

        # Execute pipeline
        results = pipeline.run(sample_data)

        # Assertions
        assert 'model' in results
        assert 'metrics' in results
        assert 'feature_names' in results
        assert results['model'].is_trained
        assert len(results['feature_names']) > 0

    def test_model_serialization_integration(self, sample_config, sample_data, tmp_path):
        """Test model and transformer serialization/deserialization."""
        # Train model
        model = BikeDemandModel(sample_config)
        processed_data = model.preprocess_data(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        model.train(X_train, y_train)

        # Save model and transformers
        model_path = tmp_path / "model.joblib"
        transformers_path = tmp_path / "transformers"
        transformers_path.mkdir()

        model.save_model(str(model_path))
        model.data_processor.save_transformers(str(transformers_path))

        # Create new instances and load
        new_model = BikeDemandModel(sample_config)
        new_processor = DataProcessor(sample_config)

        new_model.load_model(str(model_path))
        new_processor.load_transformers(str(transformers_path))

        # Test predictions are consistent
        original_predictions = model.predict(X_test)
        loaded_predictions = new_model.predict(X_test)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)

    def test_monitoring_integration(self, sample_config, sample_data):
        """Test monitoring and alerting integration."""
        # Initialize alert system
        alert_system = AlertSystem()

        # Test data quality monitoring
        processor = DataProcessor(sample_config)

        # Add some missing values to test quality checks
        test_data = sample_data.copy()
        test_data.loc[0:5, 'temp'] = np.nan  # Add missing values

        # This should trigger alerts (mocked)
        with patch.object(alert_system, 'send_alert') as mock_alert:
            # Validate data (should detect missing values)
            try:
                processor.validate_data(test_data)
            except Exception:
                pass  # Expected due to missing values

            # Check if alerts were triggered
            # Note: In real implementation, alerts would be triggered in pipeline steps

    def test_performance_monitoring(self, sample_config, sample_data):
        """Test model performance monitoring."""
        model = BikeDemandModel(sample_config)

        # Train model
        processed_data = model.preprocess_data(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_data)
        model.train(X_train, y_train)

        # Get baseline metrics
        baseline_metrics = model.evaluate(X_test, y_test)

        # Simulate performance drop by adding noise to predictions
        noisy_predictions = model.predict(X_test) + np.random.normal(0, 1000, len(X_test))

        # Calculate performance drop
        mae_drop = abs(noisy_predictions.mean() - baseline_metrics['mae']) / baseline_metrics['mae']

        # Alert system should trigger if drop exceeds threshold
        alert_system = AlertSystem()
        if mae_drop > sample_config['monitoring']['model_performance']['performance_drop_threshold']:
            with patch.object(alert_system, 'send_alert') as mock_alert:
                alert_system.alert_model_performance_drop(
                    'mae', noisy_predictions.mean(), baseline_metrics['mae']
                )
                mock_alert.assert_called()

    @pytest.mark.slow
    def test_end_to_end_pipeline_with_mlflow(self, sample_config, sample_data):
        """End-to-end test with MLflow integration."""
        with patch('mlflow.start_run') as mock_start_run, \
             patch('mlflow.log_params') as mock_log_params, \
             patch('mlflow.log_metrics') as mock_log_metrics, \
             patch('mlflow.sklearn.log_model') as mock_log_model:

            # Mock MLflow run
            mock_run = MagicMock()
            mock_start_run.return_value.__enter__.return_value = mock_run

            # Execute pipeline
            pipeline = TrainingPipeline(sample_config)
            results = pipeline.run(sample_data)

            # Verify MLflow logging was called
            assert mock_start_run.called
            assert mock_log_params.called
            assert mock_log_metrics.called
            assert mock_log_model.called

    def test_error_handling_integration(self, sample_config):
        """Test error handling across components."""
        # Test with invalid data
        invalid_data = pd.DataFrame({'invalid_col': [1, 2, 3]})

        processor = DataProcessor(sample_config)

        # Should raise ValueError for missing required columns
        with pytest.raises(ValueError, match="Missing required columns"):
            processor.validate_data(invalid_data)

        # Test model with unfitted processor
        model = BikeDemandModel(sample_config)

        with pytest.raises(ValueError, match="DataProcessor must be fitted"):
            model.preprocess_data(invalid_data)

    def test_configuration_validation(self, sample_config):
        """Test configuration validation across components."""
        # Valid config should work
        processor = DataProcessor(sample_config)
        model = BikeDemandModel(sample_config)

        assert processor.config == sample_config
        assert model.config == sample_config

        # Test with missing required config keys
        invalid_config = sample_config.copy()
        del invalid_config['features']

        with pytest.raises(KeyError):
            DataProcessor(invalid_config)

    def test_memory_efficiency(self, sample_config, sample_data):
        """Test memory efficiency of pipeline components."""
        import psutil
        import os

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Execute pipeline
        pipeline = TrainingPipeline(sample_config)
        results = pipeline.run(sample_data)

        # Check memory after execution
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (less than 500MB for this dataset)
        assert memory_increase < 500, f"Memory increase too high: {memory_increase}MB"

        # Clean up
        del results
        del pipeline
