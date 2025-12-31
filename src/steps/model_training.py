import pandas as pd
from zenml import step
from src.models.bike_demand_model import BikeDemandModel
import yaml
from pathlib import Path
import optuna
from sklearn.metrics import mean_absolute_error
from loguru import logger


@step
def model_training(df: pd.DataFrame) -> tuple[BikeDemandModel, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Train the bike demand model using the unified BikeDemandModel class with Optuna hyperparameter tuning."""
    try:
        logger.info("Starting model training step")

        # Load configuration
        config_path = Path("configs/model_config.yaml")
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Initialize model
        model = BikeDemandModel(config)

        # Preprocess data
        df_processed = model.preprocess_data(df)
        X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(df_processed)

        # Hyperparameter tuning with Optuna
        study = optuna.create_study(direction='minimize', study_name='bike_demand_rf')
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val), n_trials=50)

        # Update config with best hyperparameters
        best_params = study.best_params
        config['hyperparameters'].update(best_params)

        # Reinitialize model with best hyperparameters
        model = BikeDemandModel(config)

        # Train final model
        metrics = model.train(X_train, y_train, X_val, y_val)

        # Log to MLflow with comprehensive experiment tracking
        run_id = model.log_to_mlflow(
            metrics,
            run_name="bike_demand_training",
            experiment_name="bike_demand_model_training"
        )
        if run_id:
            logger.info(f"Model logged to MLflow with run ID: {run_id}")

        # Save model
        model_path = Path("models/bike_demand_model.joblib")
        model.save_model(str(model_path))

        logger.info(f"Model training completed. Best params: {best_params}")
        logger.info(f"Training metrics: MAE={metrics.get('train_mae', 'N/A'):.2f}, R2={metrics.get('train_r2', 'N/A'):.3f}")

        return model, X_train, X_test, y_train, y_test

    except Exception as e:
        logger.error(f"Error in model training step: {str(e)}")
        raise


def objective(trial, X_train, y_train, X_val, y_val):
    """Optuna objective function for hyperparameter tuning."""
    # Define hyperparameter search space
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 5, 30)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 10)

    # Create and train model with trial parameters
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)

    return mae
