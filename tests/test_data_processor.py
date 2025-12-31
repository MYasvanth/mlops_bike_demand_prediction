import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
from src.features.data_processor import DataProcessor


class TestDataProcessor:
    """Unit tests for DataProcessor class."""

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'features': {
                'numerical': ['temp', 'hum'],
                'categorical': ['season', 'weather']
            },
            'target': 'cnt',
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
            'dteday': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
            'temp': np.random.normal(20, 5, n_samples),
            'hum': np.random.normal(60, 10, n_samples),
            'season': np.random.choice([1, 2, 3, 4], n_samples),
            'weather': np.random.choice([1, 2, 3], n_samples),
            'cnt': np.random.normal(2000, 500, n_samples)
        }

        return pd.DataFrame(data)

    def test_processor_initialization(self, sample_config):
        """Test processor initialization."""
        processor = DataProcessor(sample_config)

        assert processor.config == sample_config
        assert not processor.is_fitted
        assert processor.scalers == {}
        assert processor.encoders == {}
        assert processor.feature_names == []

    def test_validate_data_success(self, sample_config, sample_data):
        """Test successful data validation."""
        processor = DataProcessor(sample_config)

        # Should not raise any exceptions
        processor.validate_data(sample_data)

    def test_validate_data_empty_dataframe(self, sample_config):
        """Test validation with empty DataFrame."""
        processor = DataProcessor(sample_config)
        empty_df = pd.DataFrame()

        with pytest.raises(ValueError, match="Input DataFrame is empty"):
            processor.validate_data(empty_df)

    def test_validate_data_missing_columns(self, sample_config):
        """Test validation with missing required columns."""
        processor = DataProcessor(sample_config)
        incomplete_df = pd.DataFrame({'temp': [20, 21], 'hum': [60, 61]})  # Missing categorical and target

        with pytest.raises(ValueError, match="Missing required columns"):
            processor.validate_data(incomplete_df)

    def test_preprocess_features_fit(self, sample_config, sample_data):
        """Test feature preprocessing in fit mode."""
        processor = DataProcessor(sample_config)

        processed_df = processor.preprocess_features(sample_data, fit=True)

        assert processor.is_fitted
        assert len(processor.scalers) == len(sample_config['features']['numerical'])
        assert len(processor.encoders) == len(sample_config['features']['categorical'])
        assert len(processor.feature_names) > 0

        # Check that categorical columns are encoded
        assert pd.api.types.is_numeric_dtype(processed_df['season'])
        assert pd.api.types.is_numeric_dtype(processed_df['weather'])

    def test_preprocess_features_transform(self, sample_config, sample_data):
        """Test feature preprocessing in transform mode."""
        processor = DataProcessor(sample_config)

        # Fit first
        processor.preprocess_features(sample_data, fit=True)

        # Then transform new data
        new_data = sample_data.copy()
        processed_df = processor.preprocess_features(new_data, fit=False)

        assert len(processed_df) == len(new_data)

    def test_preprocess_features_unfitted_transform(self, sample_config, sample_data):
        """Test transform without fitting."""
        processor = DataProcessor(sample_config)

        with pytest.raises(ValueError, match="Encoder for column season not fitted"):
            processor.preprocess_features(sample_data, fit=False)

    def test_handle_outliers_iqr(self, sample_config, sample_data):
        """Test outlier handling with IQR method."""
        processor = DataProcessor(sample_config)

        # Fit processor first
        processor.preprocess_features(sample_data, fit=True)

        # Handle outliers
        cleaned_df = processor.handle_outliers(sample_data, method='iqr')

        assert isinstance(cleaned_df, pd.DataFrame)
        assert len(cleaned_df) <= len(sample_data)  # May remove some rows

    def test_handle_outliers_zscore(self, sample_config, sample_data):
        """Test outlier handling with z-score method."""
        processor = DataProcessor(sample_config)

        # Fit processor first
        processor.preprocess_features(sample_data, fit=True)

        # Handle outliers
        cleaned_df = processor.handle_outliers(sample_data, method='zscore')

        assert isinstance(cleaned_df, pd.DataFrame)

    def test_handle_outliers_invalid_method(self, sample_config, sample_data):
        """Test outlier handling with invalid method."""
        processor = DataProcessor(sample_config)

        with pytest.raises(ValueError, match="Unknown outlier handling method"):
            processor.handle_outliers(sample_data, method='invalid')

    def test_create_features(self, sample_config, sample_data):
        """Test feature creation."""
        processor = DataProcessor(sample_config)

        featured_df = processor.create_features(sample_data)

        assert isinstance(featured_df, pd.DataFrame)
        assert len(featured_df) == len(sample_data)

        # Check if new features are created
        expected_new_features = ['temp_hum_interaction', 'wind_category', 'temp_season_adjusted']
        for feature in expected_new_features:
            if feature in ['temp_hum_interaction', 'temp_season_adjusted']:
                assert feature in featured_df.columns

    def test_get_feature_names_unfitted(self, sample_config):
        """Test getting feature names without fitting."""
        processor = DataProcessor(sample_config)

        with pytest.raises(ValueError, match="DataProcessor must be fitted before getting feature names"):
            processor.get_feature_names()

    def test_save_load_transformers(self, sample_config, sample_data, tmp_path):
        """Test saving and loading transformers."""
        processor = DataProcessor(sample_config)

        # Fit processor
        processor.preprocess_features(sample_data, fit=True)

        # Save transformers
        transformers_path = tmp_path / "transformers"
        processor.save_transformers(str(transformers_path))

        # Create new processor and load transformers
        new_processor = DataProcessor(sample_config)
        new_processor.load_transformers(str(transformers_path))

        assert new_processor.is_fitted
        assert len(new_processor.scalers) == len(processor.scalers)
        assert len(new_processor.encoders) == len(processor.encoders)

    @patch('joblib.dump')
    def test_save_transformers_error(self, mock_joblib_dump, sample_config, sample_data):
        """Test error handling in save_transformers."""
        processor = DataProcessor(sample_config)
        processor.preprocess_features(sample_data, fit=True)

        # Mock joblib.dump to raise an exception
        mock_joblib_dump.side_effect = Exception("Save failed")

        with pytest.raises(Exception, match="Save failed"):
            processor.save_transformers("dummy_path")

    @patch('joblib.load')
    def test_load_transformers_error(self, mock_joblib_load, sample_config):
        """Test error handling in load_transformers."""
        processor = DataProcessor(sample_config)

        # Mock joblib.load to raise an exception
        mock_joblib_load.side_effect = Exception("Load failed")

        with pytest.raises(Exception, match="Load failed"):
            processor.load_transformers("dummy_path")
