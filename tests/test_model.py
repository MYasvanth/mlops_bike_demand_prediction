import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.models.bike_demand_model import BikeDemandModel


class TestBikeDemandModel:
    """Unit tests for BikeDemandModel class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'model': {
                'name': 'test_model',
                'type': 'random_forest',
                'version': '1.0.0'
            },
            'hyperparameters': {
                'n_estimators': 10,
                'max_depth': 5,
                'random_state': 42
            },
            'features': {
                'numerical': ['temp', 'hum'],
                'categorical': ['season', 'weather']
            },
            'target': 'cnt',
            'data': {
                'train_split': 0.8,
                'val_split': 0.1,
                'test_split': 0.1,
                'random_state': 42
            },
            'evaluation': {
                'cv_folds': 3
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        }

    @pytest.fixture
    def sample_data(self):
        """Sample dataset for testing."""
        np.random.seed(42)
        n_samples = 100

        data = {
            'temp': np.random.normal(20, 5, n_samples),
            'hum': np.random.normal(60, 10, n_samples),
            'season': np.random.choice([1, 2, 3, 4], n_samples),
            'weather': np.random.choice([1, 2, 3], n_samples),
            'cnt': np.random.normal(2000, 500, n_samples)
        }

        return pd.DataFrame(data)

    def test_model_initialization(self, sample_config):
        """Test model initialization."""
        model = BikeDemandModel(sample_config)

        assert model.config == sample_config
        assert model.model is None
        assert not model.is_trained
        assert model.target_name == 'cnt'

    def test_load_data_success(self, sample_config, sample_data, tmp_path):
        """Test successful data loading."""
        model = BikeDemandModel(sample_config)

        # Save sample data to temporary file
        data_path = tmp_path / "test_data.csv"
        sample_data.to_csv(data_path, index=False)

        # Load data
        loaded_df = model.load_data(str(data_path))

        assert isinstance(loaded_df, pd.DataFrame)
        assert len(loaded_df) == len(sample_data)
        assert list(loaded_df.columns) == list(sample_data.columns)

    def test_load_data_file_not_found(self, sample_config):
        """Test data loading with non-existent file."""
        model = BikeDemandModel(sample_config)

        with pytest.raises(FileNotFoundError):
            model.load_data("non_existent_file.csv")

    def test_preprocess_data(self, sample_config, sample_data):
        """Test data preprocessing."""
        model = BikeDemandModel(sample_config)

        processed_df = model.preprocess_data(sample_data)

        assert isinstance(processed_df, pd.DataFrame)
        assert len(processed_df) == len(sample_data)
        # Check that categorical columns are encoded (should be numeric now)
        assert pd.api.types.is_numeric_dtype(processed_df['season'])
        assert pd.api.types.is_numeric_dtype(processed_df['weather'])

    def test_split_data(self, sample_config, sample_data):
        """Test data splitting."""
        model = BikeDemandModel(sample_config)

        # Preprocess first
        processed_df = model.preprocess_data(sample_data)

        # Split data
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_df)

        # Check splits
        total_samples = len(processed_df)
        expected_train = int(total_samples * 0.8)
        expected_val = int(total_samples * 0.1)
        expected_test = total_samples - expected_train - expected_val

        assert len(X_train) == expected_train
        assert len(X_val) == expected_val
        assert len(X_test) == expected_test

        # Check that all are DataFrames/Series
        assert isinstance(X_train, pd.DataFrame)
        assert isinstance(y_train, pd.Series)

    def test_train_model(self, sample_config, sample_data):
        """Test model training."""
        model = BikeDemandModel(sample_config)

        # Preprocess and split data
        processed_df = model.preprocess_data(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_df)

        # Train model
        metrics = model.train(X_train, y_train, X_val, y_val)

        assert model.is_trained
        assert isinstance(metrics, dict)
        assert 'train_mae' in metrics
        assert 'val_mae' in metrics
        assert 'cv_mae' in metrics

    def test_evaluate_model(self, sample_config, sample_data):
        """Test model evaluation."""
        model = BikeDemandModel(sample_config)

        # Train model first
        processed_df = model.preprocess_data(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_df)
        model.train(X_train, y_train)

        # Evaluate
        eval_metrics = model.evaluate(X_test, y_test)

        assert isinstance(eval_metrics, dict)
        assert 'mae' in eval_metrics
        assert 'rmse' in eval_metrics
        assert 'r2' in eval_metrics

    def test_predict_untrained_model(self, sample_config):
        """Test prediction with untrained model."""
        model = BikeDemandModel(sample_config)

        X_test = pd.DataFrame({'temp': [20], 'hum': [60], 'season': [1], 'weather': [1]})

        with pytest.raises(ValueError, match="Model must be trained before prediction"):
            model.predict(X_test)

    def test_evaluate_untrained_model(self, sample_config, sample_data):
        """Test evaluation with untrained model."""
        model = BikeDemandModel(sample_config)

        with pytest.raises(ValueError, match="Model must be trained before evaluation"):
            model.evaluate(sample_data[['temp']], sample_data['cnt'])

    def test_get_feature_importance_untrained(self, sample_config):
        """Test feature importance with untrained model."""
        model = BikeDemandModel(sample_config)

        with pytest.raises(ValueError, match="Model must be trained before getting feature importance"):
            model.get_feature_importance()

    @patch('mlflow.start_run')
    @patch('mlflow.log_params')
    @patch('mlflow.log_metrics')
    @patch('mlflow.sklearn.log_model')
    def test_log_to_mlflow(self, mock_log_model, mock_log_metrics, mock_log_params, mock_start_run, sample_config, sample_data):
        """Test MLflow logging."""
        model = BikeDemandModel(sample_config)

        # Train model first
        processed_df = model.preprocess_data(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_df)
        metrics = model.train(X_train, y_train)

        # Mock the MLflow context manager
        mock_run = MagicMock()
        mock_start_run.return_value.__enter__.return_value = mock_run

        # Log to MLflow
        model.log_to_mlflow(metrics, run_name="test_run")

        # Verify MLflow calls
        mock_start_run.assert_called_once_with(run_name="test_run")
        mock_log_params.assert_called_once_with(sample_config['hyperparameters'])
        mock_log_metrics.assert_called_once_with(metrics)
        mock_log_model.assert_called_once()

    def test_save_load_model(self, sample_config, sample_data, tmp_path):
        """Test model saving and loading."""
        model = BikeDemandModel(sample_config)

        # Train model
        processed_df = model.preprocess_data(sample_data)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(processed_df)
        model.train(X_train, y_train)

        # Save model
        model_path = tmp_path / "test_model.joblib"
        model.save_model(str(model_path))

        # Create new model instance and load
        new_model = BikeDemandModel(sample_config)
        new_model.load_model(str(model_path))

        assert new_model.is_trained
        assert new_model.model is not None

        # Test that loaded model can make predictions
        predictions = new_model.predict(X_test)
        assert len(predictions) == len(X_test)
