"""
AWS SageMaker Integration for MLOps Bike Demand Prediction

This package provides AWS SageMaker equivalents for the Azure ML components,
enabling multi-cloud ML capabilities without modifying existing Azure implementations.
"""

__version__ = "1.0.0"
__author__ = "MLOps Team"

from .sagemaker_pipeline import SageMakerPipeline
from .sagemaker_experiment import SageMakerExperimentTracker
from .sagemaker_automl import SageMakerAutoML
from .sagemaker_model import SageMakerModelManager

__all__ = [
    "SageMakerPipeline",
    "SageMakerExperimentTracker",
    "SageMakerAutoML",
    "SageMakerModelManager"
]
