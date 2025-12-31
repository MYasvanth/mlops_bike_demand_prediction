"""
Utility functions for the MLOps bike demand prediction project.
"""

import os
import json
import yaml
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional
from loguru import logger


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info(f"Configuration loaded from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config from {config_path}: {e}")
        raise


def save_json(data: Dict[str, Any], filepath: str) -> None:
    """
    Save dictionary as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Path to save file
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Data saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        raise


def load_json(filepath: str) -> Dict[str, Any]:
    """
    Load JSON file as dictionary.
    
    Args:
        filepath: Path to JSON file
        
    Returns:
        Loaded dictionary
    """
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        logger.info(f"Data loaded from {filepath}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON from {filepath}: {e}")
        raise


def save_model(model, filepath: str) -> None:
    """
    Save model using joblib.
    
    Args:
        model: Model to save
        filepath: Path to save model
    """
    try:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, filepath)
        logger.info(f"Model saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving model to {filepath}: {e}")
        raise


def load_model(filepath: str):
    """
    Load model using joblib.
    
    Args:
        filepath: Path to model file
        
    Returns:
        Loaded model
    """
    try:
        model = joblib.load(filepath)
        logger.info(f"Model loaded from {filepath}")
        return model
    except Exception as e:
        logger.error(f"Error loading model from {filepath}: {e}")
        raise


def validate_data_schema(df: pd.DataFrame, expected_columns: List[str]) -> bool:
    """
    Validate DataFrame schema.
    
    Args:
        df: DataFrame to validate
        expected_columns: List of expected column names
        
    Returns:
        True if schema is valid
    """
    try:
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            logger.error(f"Missing columns: {missing_cols}")
            return False
        
        extra_cols = set(df.columns) - set(expected_columns)
        if extra_cols:
            logger.warning(f"Extra columns found: {extra_cols}")
        
        logger.info("Data schema validation passed")
        return True
    except Exception as e:
        logger.error(f"Error validating schema: {e}")
        return False


def calculate_data_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate basic statistics for DataFrame.
    
    Args:
        df: Input DataFrame
        
    Returns:
        Dictionary with statistics
    """
    try:
        stats = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.astype(str).to_dict(),
            'numerical_stats': df.describe().to_dict(),
            'memory_usage': df.memory_usage(deep=True).sum()
        }
        logger.info("Data statistics calculated")
        return stats
    except Exception as e:
        logger.error(f"Error calculating statistics: {e}")
        raise


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None) -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
    """
    try:
        logger.remove()  # Remove default handler
        
        # Console handler
        logger.add(
            sink=lambda msg: print(msg, end=""),
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
        )
        
        # File handler if specified
        if log_file:
            Path(log_file).parent.mkdir(parents=True, exist_ok=True)
            logger.add(
                sink=log_file,
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
                rotation="10 MB"
            )
        
        logger.info("Logging setup completed")
    except Exception as e:
        print(f"Error setting up logging: {e}")


def create_feature_names(config: Dict[str, Any]) -> List[str]:
    """
    Create feature names from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of feature names
    """
    try:
        features = config.get('features', {})
        numerical = features.get('numerical', [])
        categorical = features.get('categorical', [])
        
        # Add engineered features
        engineered = [
            'temp_hum_interaction',
            'wind_category', 
            'temp_season_adjusted'
        ]
        
        all_features = numerical + categorical + engineered
        logger.info(f"Created {len(all_features)} feature names")
        return all_features
    except Exception as e:
        logger.error(f"Error creating feature names: {e}")
        raise


def validate_model_input(X: pd.DataFrame, expected_features: List[str]) -> bool:
    """
    Validate model input data.
    
    Args:
        X: Input features DataFrame
        expected_features: Expected feature names
        
    Returns:
        True if input is valid
    """
    try:
        # Check for missing features
        missing_features = set(expected_features) - set(X.columns)
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False
        
        # Check for null values
        null_counts = X.isnull().sum()
        if null_counts.any():
            logger.warning(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check data types
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                logger.error(f"Non-numeric column: {col}")
                return False
        
        logger.info("Model input validation passed")
        return True
    except Exception as e:
        logger.error(f"Error validating model input: {e}")
        return False


def format_metrics(metrics: Dict[str, float], precision: int = 4) -> Dict[str, str]:
    """
    Format metrics for display.
    
    Args:
        metrics: Dictionary of metrics
        precision: Number of decimal places
        
    Returns:
        Formatted metrics dictionary
    """
    try:
        formatted = {}
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                formatted[key] = f"{value:.{precision}f}"
            else:
                formatted[key] = str(value)
        return formatted
    except Exception as e:
        logger.error(f"Error formatting metrics: {e}")
        return metrics


def get_project_root() -> Path:
    """
    Get project root directory.
    
    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent


def ensure_directory(path: str) -> None:
    """
    Ensure directory exists.
    
    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_size(filepath: str) -> str:
    """
    Get human-readable file size.
    
    Args:
        filepath: Path to file
        
    Returns:
        Formatted file size
    """
    try:
        size = os.path.getsize(filepath)
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size < 1024.0:
                return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    except Exception as e:
        logger.error(f"Error getting file size: {e}")
        return "Unknown"