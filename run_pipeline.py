#!/usr/bin/env python3
"""
Script to run the MLOps bike demand prediction pipeline.
"""

import argparse
import sys
from pathlib import Path
from loguru import logger

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from pipelines.training_pipeline import training_pipeline


def main():
    """Main function to run the pipeline."""
    parser = argparse.ArgumentParser(description="Run MLOps Bike Demand Pipeline")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/raw/day.csv",
        help="Path to the training data CSV file"
    )
    parser.add_argument(
        "--config-path",
        type=str,
        default="configs/model_config.yaml",
        help="Path to the model configuration file"
    )

    args = parser.parse_args()

    try:
        logger.info("Starting MLOps Bike Demand Pipeline")
        logger.info(f"Data path: {args.data_path}")
        logger.info(f"Config path: {args.config_path}")

        # Validate inputs
        if not Path(args.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {args.data_path}")

        if not Path(args.config_path).exists():
            raise FileNotFoundError(f"Config file not found: {args.config_path}")

        # Run the pipeline
        logger.info("Executing training pipeline...")
        result = training_pipeline(data_path=args.data_path)

        logger.info("Pipeline execution completed successfully!")
        logger.info(f"Evaluation results: {result}")

        return 0

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
