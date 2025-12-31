from zenml import pipeline
from src.steps.data_ingestion import data_ingestion
from src.steps.data_preprocessing import data_preprocessing, data_quality_monitoring
from src.steps.model_training import model_training
from src.steps.model_evaluation import model_evaluation


@pipeline
def training_pipeline(data_path: str):
    """MLOps pipeline for training bike demand prediction model with comprehensive monitoring."""
    # Data ingestion
    df = data_ingestion(data_path)

    # Data preprocessing with quality checks
    df_processed = data_preprocessing(df)

    # Data quality monitoring (quality checks and drift detection)
    monitoring_results = data_quality_monitoring(df_processed)

    # Model training with hyperparameter tuning
    model, X_train, X_test, y_train, y_test = model_training(df_processed)

    # Model evaluation
    evaluation_report = model_evaluation(model, X_train, X_test, y_train, y_test, config={})

    return evaluation_report, monitoring_results
