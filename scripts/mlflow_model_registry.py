#!/usr/bin/env python3
"""
MLflow Model Registry Management Script

This script provides utilities for managing models in MLflow Model Registry,
including registration, versioning, staging, and deployment transitions.
"""

import argparse
import mlflow
import mlflow.sklearn
from pathlib import Path
import yaml
import logging
from typing import Optional, Dict, Any
import pandas as pd

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowModelRegistry:
    """MLflow Model Registry management class."""

    def __init__(self, tracking_uri: str = "file:./mlruns"):
        """
        Initialize MLflow Model Registry manager.

        Args:
            tracking_uri: MLflow tracking URI
        """
        mlflow.set_tracking_uri(tracking_uri)
        self.client = mlflow.tracking.MlflowClient()
        logger.info(f"Connected to MLflow tracking URI: {tracking_uri}")

    def register_model(self, run_id: str, model_name: str, model_version: str = "1.0.0") -> str:
        """
        Register a model from a specific run.

        Args:
            run_id: MLflow run ID
            model_name: Name for the registered model
            model_version: Version string for the model

        Returns:
            Model version URI
        """
        try:
            # Register the model
            model_uri = f"runs:/{run_id}/model"
            registered_model = mlflow.register_model(model_uri, model_name)

            # Add version description
            self.client.update_model_version(
                name=model_name,
                version=registered_model.version,
                description=f"Model version {model_version} registered from run {run_id}"
            )

            logger.info(f"Model registered successfully: {registered_model.name} v{registered_model.version}")
            return f"models:/{model_name}/{registered_model.version}"

        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise

    def transition_model_stage(self, model_name: str, version: str, stage: str) -> None:
        """
        Transition a model version to a different stage.

        Args:
            model_name: Registered model name
            version: Model version
            stage: Target stage (None, Staging, Production, Archived)
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            logger.info(f"Model {model_name} v{version} transitioned to {stage} stage")

        except Exception as e:
            logger.error(f"Failed to transition model stage: {e}")
            raise

    def get_latest_model_version(self, model_name: str, stage: str = "Production") -> Optional[str]:
        """
        Get the latest version of a model in a specific stage.

        Args:
            model_name: Registered model name
            stage: Stage to look for (Production, Staging, etc.)

        Returns:
            Latest version string or None if not found
        """
        try:
            versions = self.client.get_latest_versions(model_name, stages=[stage])
            if versions:
                return versions[0].version
            return None

        except Exception as e:
            logger.error(f"Failed to get latest model version: {e}")
            return None

    def load_model_for_inference(self, model_name: str, version: str = None, stage: str = "Production") -> Any:
        """
        Load a model for inference from the registry.

        Args:
            model_name: Registered model name
            version: Specific version (if None, uses latest in stage)
            stage: Stage to load from

        Returns:
            Loaded model object
        """
        try:
            if version:
                model_uri = f"models:/{model_name}/{version}"
            else:
                latest_version = self.get_latest_model_version(model_name, stage)
                if not latest_version:
                    raise ValueError(f"No model found in {stage} stage for {model_name}")
                model_uri = f"models:/{model_name}/{latest_version}"

            model = mlflow.sklearn.load_model(model_uri)
            logger.info(f"Model loaded from registry: {model_uri}")
            return model

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def compare_model_versions(self, model_name: str, version1: str, version2: str) -> Dict[str, Any]:
        """
        Compare two model versions.

        Args:
            model_name: Registered model name
            version1: First version to compare
            version2: Second version to compare

        Returns:
            Comparison results
        """
        try:
            # Get run information for both versions
            mv1 = self.client.get_model_version(model_name, version1)
            mv2 = self.client.get_model_version(model_name, version2)

            run1 = self.client.get_run(mv1.run_id)
            run2 = self.client.get_run(mv2.run_id)

            comparison = {
                "version_1": {
                    "version": version1,
                    "run_id": mv1.run_id,
                    "stage": mv1.current_stage,
                    "creation_time": mv1.creation_timestamp,
                    "metrics": run1.data.metrics
                },
                "version_2": {
                    "version": version2,
                    "run_id": mv2.run_id,
                    "stage": mv2.current_stage,
                    "creation_time": mv2.creation_timestamp,
                    "metrics": run2.data.metrics
                }
            }

            # Calculate metric differences
            common_metrics = set(run1.data.metrics.keys()) & set(run2.data.metrics.keys())
            metric_differences = {}
            for metric in common_metrics:
                diff = run2.data.metrics[metric] - run1.data.metrics[metric]
                metric_differences[metric] = {
                    "v1_value": run1.data.metrics[metric],
                    "v2_value": run2.data.metrics[metric],
                    "difference": diff,
                    "percent_change": (diff / run1.data.metrics[metric]) * 100 if run1.data.metrics[metric] != 0 else 0
                }

            comparison["metric_differences"] = metric_differences
            return comparison

        except Exception as e:
            logger.error(f"Failed to compare model versions: {e}")
            raise

    def list_registered_models(self) -> pd.DataFrame:
        """
        List all registered models with their versions and stages.

        Returns:
            DataFrame with model information
        """
        try:
            models = []
            # Get all registered model names
            registered_models = self.client.search_registered_models()

            for rm in registered_models:
                # Get all versions for each model
                versions = self.client.search_model_versions(f"name='{rm.name}'")
                for version in versions:
                    models.append({
                        "model_name": rm.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "creation_time": version.creation_timestamp,
                        "run_id": version.run_id,
                        "description": rm.description or ""
                    })

            df = pd.DataFrame(models)
            return df.sort_values(["model_name", "version"])

        except Exception as e:
            logger.error(f"Failed to list registered models: {e}")
            raise


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="MLflow Model Registry Management")
    parser.add_argument("--tracking-uri", default="file:./mlruns", help="MLflow tracking URI")

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Register model
    register_parser = subparsers.add_parser("register", help="Register a model")
    register_parser.add_argument("--run-id", required=True, help="MLflow run ID")
    register_parser.add_argument("--model-name", required=True, help="Model name")
    register_parser.add_argument("--version", default="1.0.0", help="Model version")

    # Transition stage
    stage_parser = subparsers.add_parser("stage", help="Transition model stage")
    stage_parser.add_argument("--model-name", required=True, help="Model name")
    stage_parser.add_argument("--version", required=True, help="Model version")
    stage_parser.add_argument("--stage", required=True, choices=["None", "Staging", "Production", "Archived"], help="Target stage")

    # Load model
    load_parser = subparsers.add_parser("load", help="Load model for inference")
    load_parser.add_argument("--model-name", required=True, help="Model name")
    load_parser.add_argument("--version", help="Model version (optional)")
    load_parser.add_argument("--stage", default="Production", help="Stage to load from")
    load_parser.add_argument("--output-path", help="Path to save loaded model")

    # Compare versions
    compare_parser = subparsers.add_parser("compare", help="Compare model versions")
    compare_parser.add_argument("--model-name", required=True, help="Model name")
    compare_parser.add_argument("--version1", required=True, help="First version")
    compare_parser.add_argument("--version2", required=True, help="Second version")

    # List models
    list_parser = subparsers.add_parser("list", help="List registered models")

    args = parser.parse_args()

    registry = MLflowModelRegistry(args.tracking_uri)

    if args.command == "register":
        model_uri = registry.register_model(args.run_id, args.model_name, args.version)
        print(f"Model registered: {model_uri}")

    elif args.command == "stage":
        registry.transition_model_stage(args.model_name, args.version, args.stage)
        print(f"Model {args.model_name} v{args.version} moved to {args.stage}")

    elif args.command == "load":
        model = registry.load_model_for_inference(args.model_name, args.version, args.stage)
        if args.output_path:
            import joblib
            joblib.dump(model, args.output_path)
            print(f"Model saved to: {args.output_path}")
        else:
            print("Model loaded successfully")

    elif args.command == "compare":
        comparison = registry.compare_model_versions(args.model_name, args.version1, args.version2)
        print("Model Comparison:")
        print(f"Version {args.version1}: {comparison['version_1']['metrics']}")
        print(f"Version {args.version2}: {comparison['version_2']['metrics']}")
        print(f"Differences: {comparison['metric_differences']}")

    elif args.command == "list":
        df = registry.list_registered_models()
        print(df.to_string(index=False))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
