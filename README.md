# MLOps Bike Demand Prediction

A production-ready MLOps pipeline for predicting bike rental demand using machine learning. This project implements senior-level ML engineering practices with modular architecture, configuration management, experiment tracking, and data quality monitoring.

## Features

- **Unified Model Architecture**: Centralized `BikeDemandModel` class with comprehensive ML lifecycle management
- **Modular Data Processing**: `DataProcessor` class for feature engineering, validation, and preprocessing
- **Configuration Management**: Hydra-based configuration for hyperparameters and pipeline settings
- **Hyperparameter Optimization**: Optuna integration for automated model tuning
- **Experiment Tracking**: MLflow integration for logging metrics, parameters, and models
- **Data Quality Monitoring**: Evidently integration for drift detection and data quality checks
- **Pipeline Orchestration**: ZenML for reproducible and scalable ML pipelines
- **Version Control**: DVC for data and model versioning
- **Testing**: Comprehensive unit tests for core components
- **Logging**: Structured logging with Loguru

## Project Structure

```
mlops_bike_demand/
├── configs/                    # Hydra configuration files
│   ├── model_config.yaml      # Model hyperparameters and settings
│   └── hyperparameters.yaml   # Hyperparameter search spaces
├── data/                      # Data directory
│   ├── raw/                   # Raw data files
│   ├── processed/             # Processed data
│   └── external/              # External data sources
├── dvc-pipelines/             # DVC pipeline definitions
├── models/                    # Trained models and transformers
├── mlruns/                    # MLflow experiment tracking
├── notebooks/                 # Jupyter notebooks for exploration
├── reports/                   # Evaluation reports and visualizations
│   └── figures/               # Generated plots and charts
├── scripts/                   # Utility scripts
├── src/                       # Source code
│   ├── features/              # Feature engineering modules
│   │   └── data_processor.py  # DataProcessor class
│   ├── models/                # Model definitions
│   │   └── bike_demand_model.py # BikeDemandModel class
│   ├── steps/                 # ZenML pipeline steps
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── model_training.py
│   │   └── model_evaluation.py
│   ├── pipelines/             # ZenML pipeline definitions
│   │   └── training_pipeline.py
│   └── utils.py               # Utility functions
├── tests/                     # Unit tests
│   ├── test_model.py
│   └── test_data_processor.py
├── deployment/                # Deployment configurations
├── docs/                      # Documentation
├── logs/                      # Application logs
├── .gitignore
├── dvc.yaml                   # DVC pipeline configuration
├── requirements.txt           # Python dependencies
├── setup.py                   # Package setup
├── run_pipeline.py            # Pipeline execution script
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd mlops_bike_demand
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Initialize DVC:
```bash
dvc init
```

## Usage

### Running the Pipeline

Execute the complete MLOps pipeline:

```bash
python run_pipeline.py --data-path data/raw/day.csv --config-path configs/model_config.yaml
```

### Running with DVC

Execute individual pipeline stages:

```bash
# Run data ingestion
dvc repro data_ingestion

# Run preprocessing
dvc repro data_preprocessing

# Run training
dvc repro model_training

# Run evaluation
dvc repro model_evaluation
```

### Running Tests

Execute unit tests:

```bash
pytest tests/
```

## Configuration

The pipeline is configured using Hydra. Key configuration files:

- `configs/model_config.yaml`: Main model configuration
- `configs/hyperparameters.yaml`: Hyperparameter optimization settings

## Key Components

### BikeDemandModel

Unified model class providing:
- Data loading and preprocessing
- Train/validation/test splitting
- Model training with cross-validation
- Hyperparameter optimization
- Model evaluation and metrics
- Feature importance analysis
- MLflow logging
- Model serialization

### DataProcessor

Data processing class providing:
- Data validation
- Feature preprocessing (scaling, encoding)
- Outlier handling
- Feature engineering
- Transformer serialization

### Pipeline Steps

- **Data Ingestion**: Load and validate raw data
- **Data Preprocessing**: Clean, transform, and engineer features
- **Model Training**: Train model with hyperparameter optimization
- **Model Evaluation**: Evaluate model performance and generate reports

## Monitoring and Observability

- **MLflow**: Experiment tracking and model registry
- **Evidently**: Data drift detection and quality monitoring
- **Loguru**: Structured logging throughout the pipeline

## Deployment

Deployment configurations are stored in the `deployment/` directory. Supports:
- FastAPI web service
- Docker containerization
- Cloud deployment (AWS/GCP/Azure)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with comprehensive tests
4. Ensure all tests pass
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
