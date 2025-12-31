#!/usr/bin/env python3
"""
Quick start script for MLOps Bike Demand Prediction project.
"""

import sys
import subprocess
from pathlib import Path
from loguru import logger

def run_command(command, description):
    """Run a command and handle errors."""
    try:
        logger.info(f"Running: {description}")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logger.success(f"✓ {description} completed")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed: {e}")
        if e.stdout:
            logger.error(f"STDOUT: {e.stdout}")
        if e.stderr:
            logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    """Quick start the project."""
    logger.info("🚀 MLOps Bike Demand Prediction - Quick Start")
    logger.info("=" * 60)
    
    # Check if data exists
    data_file = Path("data/raw/day.csv")
    if not data_file.exists():
        logger.error(f"Data file not found: {data_file}")
        logger.info("Please ensure the bike sharing dataset is in data/raw/day.csv")
        return False
    
    # Setup project
    if not run_command("python setup_project.py", "Project setup"):
        return False
    
    # Run the pipeline
    if not run_command("python run_pipeline.py", "ML Pipeline execution"):
        return False
    
    logger.success("🎉 Quick start completed successfully!")
    logger.info("\nNext steps:")
    logger.info("1. View results in reports/ directory")
    logger.info("2. Start MLflow UI: mlflow ui")
    logger.info("3. Check model artifacts in models/ directory")
    logger.info("4. Run tests: pytest tests/")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)