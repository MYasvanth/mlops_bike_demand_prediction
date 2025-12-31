# MLOps Bike Demand Prediction Makefile

.PHONY: help setup install clean train evaluate pipeline test lint format docker

# Default target
help:
	@echo "Available commands:"
	@echo "  setup      - Setup project environment"
	@echo "  install    - Install dependencies"
	@echo "  clean      - Clean generated files"
	@echo "  train      - Train the model"
	@echo "  evaluate   - Evaluate the model"
	@echo "  pipeline   - Run full pipeline"
	@echo "  test       - Run tests"
	@echo "  lint       - Run code linting"
	@echo "  format     - Format code"
	@echo "  docker     - Build Docker image"
	@echo "  mlflow     - Start MLflow UI"
	@echo "  zenml      - Start ZenML dashboard"

# Setup project
setup:
	python setup_project.py

# Install dependencies
install:
	pip install -r requirements.txt

# Clean generated files
clean:
	rm -rf models/*.joblib
	rm -rf models/transformers/*
	rm -rf data/processed/*
	rm -rf reports/*
	rm -rf logs/*
	rm -rf __pycache__
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

# Train model
train:
	python -c "import sys; sys.path.append('src'); from models.bike_demand_model import BikeDemandModel; from utils import load_config; import pandas as pd; config = load_config('configs/model_config.yaml'); model = BikeDemandModel(config); df = pd.read_csv('data/raw/day.csv'); df_processed = model.preprocess_data(df); X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(df_processed); metrics = model.train(X_train, y_train, X_val, y_val); model.save_model('models/bike_demand_model.joblib'); print(f'Training completed - MAE: {metrics.get(\"train_mae\", 0):.2f}')"

# Evaluate model
evaluate:
	python -c "import sys; sys.path.append('src'); from models.bike_demand_model import BikeDemandModel; from utils import load_config; import pandas as pd; config = load_config('configs/model_config.yaml'); model = BikeDemandModel(config); model.load_model('models/bike_demand_model.joblib'); df = pd.read_csv('data/raw/day.csv'); df_processed = model.preprocess_data(df); X_train, X_val, X_test, y_train, y_val, y_test = model.split_data(df_processed); metrics = model.evaluate(X_test, y_test); print(f'Evaluation completed - MAE: {metrics.get(\"mae\", 0):.2f}')"

# Run full pipeline
pipeline:
	python run_pipeline.py --data-path data/raw/day.csv --config-path configs/model_config.yaml

# Run DVC pipeline
dvc-pipeline:
	dvc repro

# Run tests
test:
	pytest tests/ -v

# Run linting
lint:
	flake8 src/ tests/ --max-line-length=120
	pylint src/ --disable=C0114,C0115,C0116

# Format code
format:
	black src/ tests/ --line-length=120
	isort src/ tests/

# Build Docker image
docker:
	docker build -t bike-demand-prediction .

# Start MLflow UI
mlflow:
	mlflow ui --host 0.0.0.0 --port 5000

# Start ZenML dashboard
zenml:
	zenml up

# Quick start (setup + install + pipeline)
quickstart: setup install pipeline
	@echo "Quick start completed!"

# Development setup
dev-setup: setup install
	pip install black flake8 pylint isort pytest
	@echo "Development environment ready!"