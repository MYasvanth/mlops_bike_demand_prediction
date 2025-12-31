# MLOps Bike Demand Prediction - Implementation Guide

## 🚀 Quick Start

### Option 1: One-Command Setup
```bash
python quick_start.py
```

### Option 2: Step-by-Step Setup
```bash
# 1. Setup project environment
python setup_project.py

# 2. Run the ML pipeline
python run_pipeline.py

# 3. Start MLflow UI (optional)
mlflow ui
```

### Option 3: Using Make Commands
```bash
# Quick start with all setup
make quickstart

# Or individual commands
make setup
make pipeline
make mlflow
```

## 📋 Prerequisites

- Python 3.8+
- Git
- Optional: Docker, DVC

## 🏗️ Project Architecture

```
mlops_bike_demand/
├── 📊 data/                    # Data storage
│   ├── raw/                   # Original datasets
│   ├── processed/             # Processed data
│   └── external/              # External data sources
├── ⚙️ configs/                 # Configuration files
│   ├── model_config.yaml     # Model parameters
│   └── hyperparameters.yaml  # Hyperparameter spaces
├── 🤖 src/                     # Source code
│   ├── models/               # Model definitions
│   ├── features/             # Feature engineering
│   ├── steps/                # Pipeline steps
│   ├── pipelines/            # ZenML pipelines
│   ├── api/                  # API endpoints
│   ├── monitoring/           # Monitoring & alerts
│   └── visualization/        # Model explainability
├── 🧪 tests/                   # Unit tests
├── 📈 reports/                 # Generated reports
├── 🏭 models/                  # Trained models
├── 📋 logs/                    # Application logs
└── 🐳 deployment/              # Deployment configs
```

## 🔧 Core Components

### 1. **BikeDemandModel** (`src/models/bike_demand_model.py`)
Unified model class with:
- Data loading & preprocessing
- Train/validation/test splitting
- Model training with cross-validation
- Hyperparameter optimization (Optuna)
- Model evaluation & metrics
- Feature importance analysis
- MLflow logging
- Model serialization

### 2. **DataProcessor** (`src/features/data_processor.py`)
Data processing class with:
- Data validation & quality checks
- Feature preprocessing (scaling, encoding)
- Outlier handling (IQR method)
- Feature engineering
- Data drift detection (Evidently)
- Transformer serialization

### 3. **Pipeline Steps** (`src/steps/`)
- **Data Ingestion**: Load and validate raw data
- **Data Preprocessing**: Clean, transform, and engineer features
- **Model Training**: Train with hyperparameter optimization
- **Model Evaluation**: Evaluate performance and generate reports

### 4. **Monitoring & Alerting** (`src/monitoring/`)
- Data quality monitoring
- Data drift detection
- Model performance monitoring
- Email alerting system

### 5. **Model Explainability** (`src/visualization/`)
- SHAP explanations (global & local)
- LIME explanations
- Partial dependence plots
- Feature interaction analysis

## 🚀 Usage Examples

### Training a Model
```python
from src.models.bike_demand_model import BikeDemandModel
from src.utils import load_config
import pandas as pd

# Load configuration
config = load_config('configs/model_config.yaml')

# Initialize model
model = BikeDemandModel(config)

# Load and preprocess data
df = pd.read_csv('data/raw/day.csv')
df_processed = model.preprocess_data(df)

# Split data
X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(df_processed)

# Train model
metrics = model.train(X_train, y_train, X_val, y_val)

# Save model
model.save_model('models/bike_demand_model.joblib')
```

### Making Predictions
```python
# Load trained model
model = BikeDemandModel(config)
model.load_model('models/bike_demand_model.joblib')

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
test_metrics = model.evaluate(X_test, y_test)
```

### Data Processing
```python
from src.features.data_processor import DataProcessor

# Initialize processor
processor = DataProcessor(config)

# Validate data
processor.validate_data(df)

# Preprocess features
df_processed = processor.preprocess_features(df, fit=True)

# Handle outliers
df_clean = processor.handle_outliers(df_processed)

# Create additional features
df_featured = processor.create_features(df_clean)

# Save transformers
processor.save_transformers('models/transformers')
```

## 🌐 API Usage

### Start the API Server
```bash
# Using Python
python src/api/fastapi_app.py

# Using uvicorn
uvicorn src.api.fastapi_app:app --host 0.0.0.0 --port 8000
```

### Make Predictions via API
```python
import requests

# Single prediction
data = {
    "season": 2,
    "yr": 1,
    "mnth": 6,
    "holiday": 0,
    "weekday": 1,
    "workingday": 1,
    "weathersit": 1,
    "temp": 0.6,
    "atemp": 0.58,
    "hum": 0.63,
    "windspeed": 0.15
}

response = requests.post("http://localhost:8000/predict", json=data)
prediction = response.json()
print(f"Predicted demand: {prediction['prediction']}")
```

## 📊 Monitoring & Observability

### MLflow Tracking
```bash
# Start MLflow UI
mlflow ui --host 0.0.0.0 --port 5000

# View at: http://localhost:5000
```

### Data Quality Monitoring
```python
from src.features.data_processor import DataProcessor

processor = DataProcessor(config)

# Check data quality
quality_results = processor.check_data_quality(df)

# Detect data drift
drift_results = processor.detect_data_drift(reference_data, current_data)
```

### Model Performance Monitoring
```python
# Monitor model performance
performance_results = model.monitor_performance(X_test, y_test)

if performance_results['performance_drop']:
    print("⚠️ Model performance degradation detected!")
```

## 🧪 Testing

### Run All Tests
```bash
pytest tests/ -v
```

### Run Specific Test Categories
```bash
# Unit tests only
pytest tests/ -m unit

# Integration tests
pytest tests/ -m integration

# End-to-end tests
pytest tests/ -m e2e
```

## 🐳 Docker Deployment

### Build Docker Image
```bash
docker build -t bike-demand-prediction .
```

### Run Container
```bash
# Run pipeline
docker run bike-demand-prediction

# Run API server
docker run -p 8000:8000 bike-demand-prediction python src/api/fastapi_app.py
```

## 📈 DVC Pipeline

### Run DVC Pipeline
```bash
# Run entire pipeline
dvc repro

# Run specific stage
dvc repro model_training

# View pipeline
dvc dag
```

### Track Data & Models
```bash
# Add data to DVC
dvc add data/raw/day.csv

# Add model to DVC
dvc add models/bike_demand_model.joblib

# Commit changes
git add .
git commit -m "Add data and model tracking"
```

## 🔧 Configuration

### Model Configuration (`configs/model_config.yaml`)
```yaml
model:
  name: "bike_demand_rf_regressor"
  type: "random_forest"
  version: "1.0.0"

hyperparameters:
  n_estimators: 100
  max_depth: null
  min_samples_split: 2
  min_samples_leaf: 1
  random_state: 42
  n_jobs: -1

features:
  numerical: ["temp", "atemp", "hum", "windspeed"]
  categorical: ["season", "yr", "mnth", "holiday", "weekday", "workingday", "weathersit"]

target: "cnt"

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  random_state: 42
```

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Missing Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Data File Not Found**
   - Ensure `data/raw/day.csv` exists
   - Download from: [Bike Sharing Dataset](https://archive.ics.uci.edu/ml/datasets/bike+sharing+dataset)

4. **MLflow Tracking Issues**
   ```bash
   mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
   ```

### Performance Optimization

1. **Memory Usage**
   - Use data sampling for large datasets
   - Enable garbage collection
   - Use efficient data types

2. **Training Speed**
   - Adjust `n_jobs` parameter
   - Use feature selection
   - Reduce hyperparameter search space

3. **Prediction Latency**
   - Cache preprocessed transformers
   - Use model quantization
   - Implement batch prediction

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [ZenML Documentation](https://docs.zenml.io/)
- [DVC Documentation](https://dvc.org/doc)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Evidently Documentation](https://docs.evidentlyai.com/)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes with comprehensive tests
4. Run tests: `pytest tests/`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.