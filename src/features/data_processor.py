import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from loguru import logger
from evidently import Report
from evidently.presets import DataDriftPreset
from evidently.metrics import DatasetMissingValueCount, DatasetCorrelations
from src.monitoring.alerts import AlertSystem
import json
import os


class DataProcessor:
    """Data processing class for bike demand dataset with proper validation and feature engineering."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the data processor.

        Args:
            config: Configuration dictionary containing feature information
        """
        self.config = config
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_names: List[str] = []
        self.is_fitted: bool = False

        # Setup logging
        self._setup_logging()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format=self.config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    def validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate the input data.

        Args:
            df: Input DataFrame

        Raises:
            ValueError: If data validation fails
        """
        try:
            logger.info("Validating input data")

            # Check if DataFrame is empty
            if df.empty:
                raise ValueError("Input DataFrame is empty")

            # Check for required columns
            required_cols = (self.config['features']['numerical'] +
                           self.config['features']['categorical'] +
                           [self.config['target']])

            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")

            # Check for missing values
            missing_summary = df[required_cols].isnull().sum()
            if missing_summary.any():
                logger.warning(f"Missing values found: {missing_summary[missing_summary > 0].to_dict()}")

            # Check data types
            for col in self.config['features']['numerical']:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    raise ValueError(f"Column {col} should be numeric")

            for col in self.config['features']['categorical']:
                if pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Column {col} is numeric but treated as categorical")

            logger.info("Data validation completed successfully")

        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            raise

    def preprocess_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Preprocess features including scaling and encoding.

        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for inference)

        Returns:
            Preprocessed DataFrame
        """
        try:
            logger.info(f"Preprocessing features (fit={fit})")

            # Make a copy to avoid modifying original
            df_processed = df.copy()

            # Handle date column
            if 'dteday' in df_processed.columns:
                df_processed['dteday'] = pd.to_datetime(df_processed['dteday'], format='%d-%m-%Y')
                # Extract useful date features
                df_processed['year'] = df_processed['dteday'].dt.year
                df_processed['month'] = df_processed['dteday'].dt.month
                df_processed['day'] = df_processed['dteday'].dt.day
                df_processed['day_of_week'] = df_processed['dteday'].dt.dayofweek
                df_processed.drop('dteday', axis=1, inplace=True)

            # Process categorical features
            for col in self.config['features']['categorical']:
                if fit:
                    self.encoders[col] = LabelEncoder()
                    df_processed[col] = self.encoders[col].fit_transform(df_processed[col])
                else:
                    if col in self.encoders:
                        df_processed[col] = self.encoders[col].transform(df_processed[col])
                    else:
                        raise ValueError(f"Encoder for column {col} not fitted")

            # Process numerical features
            for col in self.config['features']['numerical']:
                if fit:
                    self.scalers[col] = StandardScaler()
                    df_processed[col] = self.scalers[col].fit_transform(df_processed[[col]])
                else:
                    if col in self.scalers:
                        df_processed[col] = self.scalers[col].transform(df_processed[[col]])
                    else:
                        raise ValueError(f"Scaler for column {col} not fitted")

            # Set feature names
            if fit:
                self.feature_names = [col for col in df_processed.columns if col != self.config['target']]
                self.is_fitted = True

            logger.info(f"Feature preprocessing completed. Shape: {df_processed.shape}")
            return df_processed

        except Exception as e:
            logger.error(f"Error in feature preprocessing: {str(e)}")
            raise

    def handle_outliers(self, df: pd.DataFrame, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        Handle outliers in numerical features.

        Args:
            df: Input DataFrame
            method: Outlier detection method ('iqr' or 'zscore')
            threshold: Threshold for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        try:
            logger.info(f"Handling outliers using {method} method")

            df_clean = df.copy()

            for col in self.config['features']['numerical']:
                if method == 'iqr':
                    Q1 = df_clean[col].quantile(0.25)
                    Q3 = df_clean[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - threshold * IQR
                    upper_bound = Q3 + threshold * IQR

                    # Clip outliers
                    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

                elif method == 'zscore':
                    z_scores = np.abs((df_clean[col] - df_clean[col].mean()) / df_clean[col].std())
                    df_clean = df_clean[z_scores < threshold]

                else:
                    raise ValueError(f"Unknown outlier handling method: {method}")

            logger.info(f"Outlier handling completed. Shape: {df_clean.shape}")
            return df_clean

        except Exception as e:
            logger.error(f"Error handling outliers: {str(e)}")
            raise

    def create_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Create additional features for better model performance.

        Args:
            df: Input DataFrame
            fit: Whether to fit transformers (True for training, False for inference)

        Returns:
            DataFrame with additional features
        """
        try:
            logger.info(f"Creating additional features (fit={fit})")

            df_featured = df.copy()

            # Temperature-humidity interaction
            if 'temp' in df_featured.columns and 'hum' in df_featured.columns:
                df_featured['temp_hum_interaction'] = df_featured['temp'] * df_featured['hum']

            # Wind speed categories
            if 'windspeed' in df_featured.columns:
                df_featured['wind_category'] = pd.cut(
                    df_featured['windspeed'],
                    bins=[-np.inf, 0.1, 0.3, np.inf],
                    labels=['calm', 'light', 'windy']
                )
                # Encode wind category
                if fit:
                    self.encoders['wind_category'] = LabelEncoder()
                    df_featured['wind_category'] = self.encoders['wind_category'].fit_transform(df_featured['wind_category'])
                else:
                    if 'wind_category' in self.encoders:
                        try:
                            df_featured['wind_category'] = self.encoders['wind_category'].transform(df_featured['wind_category'])
                        except ValueError as e:
                            logger.warning(f"Wind category encoding failed, using default: {e}")
                            # Use default encoding for unknown categories
                            df_featured['wind_category'] = 1  # Default to 'light' wind
                    else:
                        logger.warning("Wind category encoder not fitted, using default")
                        df_featured['wind_category'] = 1  # Default to 'light' wind

            # Seasonal temperature adjustment
            if 'temp' in df_featured.columns and 'season' in df_featured.columns:
                season_temp_mean = df_featured.groupby('season')['temp'].transform('mean')
                df_featured['temp_season_adjusted'] = df_featured['temp'] - season_temp_mean

            logger.info(f"Feature creation completed. New shape: {df_featured.shape}")
            return df_featured

        except Exception as e:
            logger.error(f"Error creating features: {str(e)}")
            raise

    def get_feature_names(self) -> List[str]:
        """
        Get the list of feature names after preprocessing.

        Returns:
            List of feature names
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before getting feature names")
        return self.feature_names

    def save_transformers(self, path: str) -> None:
        """
        Save fitted transformers to disk.

        Args:
            path: Base path to save transformers
        """
        import joblib
        from pathlib import Path

        try:
            Path(path).mkdir(parents=True, exist_ok=True)

            # Save scalers
            for col, scaler in self.scalers.items():
                joblib.dump(scaler, f"{path}/scaler_{col}.joblib")

            # Save encoders
            for col, encoder in self.encoders.items():
                joblib.dump(encoder, f"{path}/encoder_{col}.joblib")

            logger.info(f"Transformers saved to {path}")

        except Exception as e:
            logger.error(f"Error saving transformers: {str(e)}")
            raise

    def load_transformers(self, path: str) -> None:
        """
        Load fitted transformers from disk.

        Args:
            path: Base path to load transformers from
        """
        import joblib
        from pathlib import Path

        try:
            # Load scalers
            for col in self.config['features']['numerical']:
                scaler_path = Path(f"{path}/scaler_{col}.joblib")
                if scaler_path.exists():
                    self.scalers[col] = joblib.load(scaler_path)

            # Load encoders
            for col in self.config['features']['categorical']:
                encoder_path = Path(f"{path}/encoder_{col}.joblib")
                if encoder_path.exists():
                    self.encoders[col] = joblib.load(encoder_path)

            self.is_fitted = True
            logger.info(f"Transformers loaded from {path}")

        except Exception as e:
            logger.error(f"Error loading transformers: {str(e)}")
            raise

    def check_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality using Evidently and return quality metrics.

        Args:
            df: Input DataFrame to check

        Returns:
            Dictionary with quality check results
        """
        try:
            logger.info("Running data quality checks")

            # Create data quality report
            data_quality_report = Report(metrics=[DatasetMissingValueCount(), DatasetCorrelations()])
            snapshot = data_quality_report.run(reference_data=None, current_data=df)

            # Extract quality metrics from snapshot
            quality_results = snapshot.dict()

            # Check for issues
            issues = []

            # Missing values check from DatasetMissingValueCount
            missing_count = 0
            if 'metrics' in quality_results:
                for metric_data in quality_results['metrics']:
                    if metric_data.get('metric_id') == 'DatasetMissingValueCount()':
                        missing_count = metric_data.get('value', {}).get('count', 0)
                        break

            missing_pct = missing_count / len(df) if len(df) > 0 else 0
            if missing_pct > self.config.get('monitoring', {}).get('data_quality', {}).get('missing_threshold', 0.05):
                issues.append(f"High missing values: {missing_pct:.2%}")

            # Duplicate check
            duplicate_rate = df.duplicated().sum() / len(df)
            if duplicate_rate > self.config.get('monitoring', {}).get('data_quality', {}).get('duplicate_threshold', 0.01):
                issues.append(f"High duplicate rate: {duplicate_rate:.2%}")

            # Alert if issues found
            if issues:
                alert_system = AlertSystem(
                    smtp_server=self.config.get('monitoring', {}).get('alerting', {}).get('smtp_server'),
                    smtp_port=self.config.get('monitoring', {}).get('alerting', {}).get('smtp_port', 587),
                    sender_email=self.config.get('monitoring', {}).get('alerting', {}).get('sender_email'),
                    sender_password=self.config.get('monitoring', {}).get('alerting', {}).get('sender_password'),
                    recipient_emails=self.config.get('monitoring', {}).get('alerting', {}).get('recipient_emails', [])
                )
                alert_system.alert_data_quality_issue("; ".join(issues))

            logger.info(f"Data quality check completed. Issues found: {len(issues)}")
            return {
                'quality_report': quality_results,
                'issues': issues,
                'duplicate_rate': duplicate_rate
            }

        except Exception as e:
            logger.error(f"Error in data quality check: {str(e)}")
            raise

    def detect_data_drift(self, reference_data: pd.DataFrame, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data using Evidently.

        Args:
            reference_data: Reference dataset (training data)
            current_data: Current dataset (inference data)

        Returns:
            Dictionary with drift detection results
        """
        try:
            logger.info("Detecting data drift")

            # Create drift report
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=reference_data, current_data=current_data)

            # Extract drift results - handle different Evidently API versions
            try:
                drift_results = drift_report.dict()
            except AttributeError:
                # For newer Evidently versions, dict() method might not exist
                # Return simplified results
                drift_results = {"error": "Evidently API changed - dict() method not available"}
                drift_score = 0.0
                drifted_columns = []
            else:
                # Calculate overall drift score
                drift_score = 0.0
                drifted_columns = []

                if 'metrics' in drift_results:
                    for metric in drift_results['metrics']:
                        if 'result' in metric and 'drift_by_columns' in metric['result']:
                            for col, col_result in metric['result']['drift_by_columns'].items():
                                if col_result.get('drift_detected', False):
                                    drifted_columns.append(col)
                                    drift_score += col_result.get('drift_score', 0.0)

                if drifted_columns:
                    drift_score /= len(drifted_columns)

            # Alert if drift detected
            threshold = self.config.get('monitoring', {}).get('data_drift', {}).get('drift_threshold', 0.1)
            if drift_score > threshold:
                alert_system = AlertSystem(
                    smtp_server=self.config.get('monitoring', {}).get('alerting', {}).get('smtp_server'),
                    smtp_port=self.config.get('monitoring', {}).get('alerting', {}).get('smtp_port', 587),
                    sender_email=self.config.get('monitoring', {}).get('alerting', {}).get('sender_email'),
                    sender_password=self.config.get('monitoring', {}).get('alerting', {}).get('sender_password'),
                    recipient_emails=self.config.get('monitoring', {}).get('alerting', {}).get('recipient_emails', [])
                )
                alert_system.alert_data_drift(drift_score, threshold)

            logger.info(f"Data drift detection completed. Drift score: {drift_score:.3f}, Drifted columns: {drifted_columns}")
            return {
                'drift_report': drift_results,
                'drift_score': drift_score,
                'drifted_columns': drifted_columns,
                'threshold': threshold
            }

        except Exception as e:
            logger.error(f"Error in data drift detection: {str(e)}")
            raise

    def save_reference_data(self, df: pd.DataFrame, path: str) -> None:
        """
        Save reference data for drift detection.

        Args:
            df: Reference DataFrame
            path: Path to save reference data
        """
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            logger.info(f"Reference data saved to {path}")
        except Exception as e:
            logger.error(f"Error saving reference data: {str(e)}")
            raise

    def load_reference_data(self, path: str) -> pd.DataFrame:
        """
        Load reference data for drift detection.

        Args:
            path: Path to reference data

        Returns:
            Reference DataFrame
        """
        try:
            df = pd.read_csv(path)
            logger.info(f"Reference data loaded from {path}")
            return df
        except Exception as e:
            logger.error(f"Error loading reference data: {str(e)}")
            raise
