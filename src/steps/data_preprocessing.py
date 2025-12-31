import pandas as pd
from zenml import step
from src.features.data_processor import DataProcessor
import yaml
from pathlib import Path
from loguru import logger
import mlflow


@step
def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the bike demand dataset using the unified DataProcessor class with quality monitoring."""
    try:
        logger.info("Starting data preprocessing step")

        # Load configuration
        config_path = Path("configs/model_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize data processor
        processor = DataProcessor(config)

        # Run data quality checks if enabled
        if config.get('monitoring', {}).get('data_quality', {}).get('enable_checks', True):
            quality_results = processor.check_data_quality(df)
            logger.info(f"Data quality check results: {len(quality_results['issues'])} issues found")

            # Log quality metrics to MLflow
            with mlflow.start_run(run_name="data_quality_check", nested=True):
                mlflow.log_metric("duplicate_rate", quality_results['duplicate_rate'])
                mlflow.log_param("quality_issues_count", len(quality_results['issues']))
                if quality_results['issues']:
                    mlflow.log_param("quality_issues", "; ".join(quality_results['issues']))

        # Validate data
        processor.validate_data(df)

        # Preprocess features (fit=True for training)
        df_processed = processor.preprocess_features(df, fit=True)

        # Handle outliers
        df_processed = processor.handle_outliers(df_processed, method='iqr')

        # Create additional features
        df_processed = processor.create_features(df_processed)

        # Save transformers for later use
        transformers_path = Path("models/transformers")
        processor.save_transformers(str(transformers_path))

        # Save reference data for drift detection if enabled
        if config.get('monitoring', {}).get('data_drift', {}).get('enable_detection', True):
            reference_path = config.get('monitoring', {}).get('data_drift', {}).get('reference_data_path', 'data/processed/reference_data.csv')
            processor.save_reference_data(df_processed, reference_path)
            logger.info(f"Reference data saved for drift detection at {reference_path}")

        logger.info(f"Data preprocessing completed. Final shape: {df_processed.shape}")
        return df_processed

    except Exception as e:
        logger.error(f"Error in data preprocessing step: {str(e)}")
        raise


@step
def data_quality_monitoring(df: pd.DataFrame) -> dict:
    """Monitor data quality and detect drift in the preprocessing pipeline."""
    try:
        logger.info("Starting data quality monitoring step")

        # Load configuration
        config_path = Path("configs/model_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize data processor
        processor = DataProcessor(config)

        # Run data quality checks
        quality_results = processor.check_data_quality(df)

        # Check for data drift if reference data exists
        drift_results = None
        if config.get('monitoring', {}).get('data_drift', {}).get('enable_detection', True):
            reference_path = config.get('monitoring', {}).get('data_drift', {}).get('reference_data_path', 'data/processed/reference_data.csv')
            if Path(reference_path).exists():
                reference_data = processor.load_reference_data(reference_path)
                drift_results = processor.detect_data_drift(reference_data, df)
                logger.info(f"Data drift detection completed. Drift score: {drift_results['drift_score']:.3f}")

        # Log monitoring results to MLflow
        with mlflow.start_run(run_name="data_monitoring", nested=True):
            mlflow.log_metric("duplicate_rate", quality_results['duplicate_rate'])
            mlflow.log_param("quality_issues_count", len(quality_results['issues']))

            if quality_results['issues']:
                mlflow.log_param("quality_issues", "; ".join(quality_results['issues']))

            if drift_results:
                mlflow.log_metric("drift_score", drift_results['drift_score'])
                mlflow.log_param("drifted_columns", ", ".join(drift_results['drifted_columns']))
                mlflow.log_metric("drift_threshold", drift_results['threshold'])

        monitoring_results = {
            'quality_results': quality_results,
            'drift_results': drift_results
        }

        logger.info("Data quality monitoring completed")
        return monitoring_results

    except Exception as e:
        logger.error(f"Error in data quality monitoring step: {str(e)}")
        raise
