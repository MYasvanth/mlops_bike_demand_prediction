#!/usr/bin/env python3
"""
Setup script for MLOps Bike Demand Prediction project.
"""

import os
import subprocess
import sys
from pathlib import Path

def create_directories():
    """Create necessary project directories."""
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models/transformers",
        "reports/figures",
        "reports/explainability/plots",
        "logs",
        "notebooks",
        "tests",
        "deployment"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created directory: {directory}")

def install_dependencies():
    """Install required Python packages."""
    print("Installing dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error installing dependencies: {e}")
        return False
    return True

def initialize_mlflow():
    """Initialize MLflow tracking."""
    print("Initializing MLflow...")
    try:
        import mlflow
        mlflow.set_tracking_uri("file:./mlruns")
        print("✓ MLflow initialized")
    except ImportError:
        print("✗ MLflow not installed")
        return False
    return True

def initialize_dvc():
    """Initialize DVC for data versioning."""
    print("Initializing DVC...")
    try:
        if not Path(".dvc").exists():
            subprocess.check_call(["dvc", "init"])
            print("✓ DVC initialized")
        else:
            print("✓ DVC already initialized")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ DVC not available or error occurred")
        return False
    return True

def setup_zenml():
    """Initialize ZenML for pipeline orchestration."""
    print("Setting up ZenML...")
    try:
        subprocess.check_call(["zenml", "init"])
        print("✓ ZenML initialized")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ ZenML not available or error occurred")
        return False
    return True

def main():
    """Main setup function."""
    print("🚀 Setting up MLOps Bike Demand Prediction Project")
    print("=" * 50)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Setup failed at dependency installation")
        return
    
    # Initialize tools
    initialize_mlflow()
    initialize_dvc()
    setup_zenml()
    
    print("\n" + "=" * 50)
    print("✅ Project setup completed successfully!")
    print("\nNext steps:")
    print("1. Run the pipeline: python run_pipeline.py")
    print("2. View MLflow UI: mlflow ui")
    print("3. Check ZenML dashboard: zenml up")

if __name__ == "__main__":
    main()