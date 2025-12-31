import logging
from typing import Dict, Any, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import uuid
from collections import defaultdict
import mlflow
import mlflow.sklearn
from loguru import logger


class ExperimentTracker:
    """Advanced experiment tracking system for ML model development and comparison."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the experiment tracker.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.experiments = {}
        self.active_experiments = {}

        # Setup logging
        self._setup_logging()

        # Initialize tracking storage
        self._init_storage()

    def _setup_logging(self) -> None:
        """Setup logging configuration."""
        logging.basicConfig(
            level=getattr(logging, self.config.get('logging', {}).get('level', 'INFO')),
            format=self.config.get('logging', {}).get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        )

    def _init_storage(self) -> None:
        """Initialize experiment storage."""
        self.experiments_dir = Path("experiments")
        self.experiments_dir.mkdir(exist_ok=True)

        self.metadata_file = self.experiments_dir / "experiments_metadata.json"
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.experiments = json.load(f)

    def start_experiment(
        self,
        experiment_name: str,
        description: str = "",
        tags: Dict[str, Any] = None,
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Start a new experiment.

        Args:
            experiment_name: Name of the experiment
            description: Description of the experiment
            tags: Tags for categorization
            metadata: Additional metadata

        Returns:
            Experiment ID
        """
        try:
            experiment_id = str(uuid.uuid4())

            experiment = {
                'experiment_id': experiment_id,
                'experiment_name': experiment_name,
                'description': description,
                'status': 'active',
                'created_at': datetime.now().isoformat(),
                'tags': tags or {},
                'metadata': metadata or {},
                'runs': [],
                'summary': {}
            }

            self.experiments[experiment_id] = experiment
            self.active_experiments[experiment_id] = experiment

            # Save metadata
            self._save_metadata()

            # Start MLflow experiment
            mlflow.set_experiment(experiment_name)

            logger.info(f"Started experiment: {experiment_name} (ID: {experiment_id})")
            return experiment_id

        except Exception as e:
            logger.error(f"Error starting experiment: {str(e)}")
            raise

    def start_run(
        self,
        experiment_id: str,
        run_name: str,
        parameters: Dict[str, Any] = None,
        tags: Dict[str, Any] = None
    ) -> str:
        """
        Start a new run within an experiment.

        Args:
            experiment_id: ID of the parent experiment
            run_name: Name of the run
            parameters: Run parameters
            tags: Run tags

        Returns:
            Run ID
        """
        try:
            if experiment_id not in self.active_experiments:
                raise ValueError(f"Experiment {experiment_id} is not active")

            run_id = str(uuid.uuid4())

            run = {
                'run_id': run_id,
                'run_name': run_name,
                'experiment_id': experiment_id,
                'status': 'running',
                'started_at': datetime.now().isoformat(),
                'parameters': parameters or {},
                'tags': tags or {},
                'metrics': {},
                'artifacts': {},
                'logs': []
            }

            # Add run to experiment
            self.experiments[experiment_id]['runs'].append(run)

            # Start MLflow run
            with mlflow.start_run(run_name=run_name):
                mlflow.set_tags(tags or {})
                if parameters:
                    for key, value in parameters.items():
                        mlflow.log_param(key, value)

                # Get MLflow run ID and store it
                mlflow_run = mlflow.active_run()
                mlflow_run_id = mlflow_run.info.run_id
                run['mlflow_run_id'] = mlflow_run_id

                # Store custom run ID in MLflow
                mlflow.set_tag("custom_run_id", run_id)
                mlflow.set_tag("experiment_id", experiment_id)

            logger.info(f"Started run: {run_name} (ID: {run_id}) in experiment {experiment_id}")
            return run_id

        except Exception as e:
            logger.error(f"Error starting run: {str(e)}")
            raise

    def log_metric(self, run_id: str, key: str, value: float, step: int = None) -> None:
        """
        Log a metric for a run.

        Args:
            run_id: Run ID
            key: Metric key
            value: Metric value
            step: Step number (for time series metrics)
        """
        try:
            # Find the run
            run = self._find_run(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")

            # Log to internal storage
            if key not in run['metrics']:
                run['metrics'][key] = []

            run['metrics'][key].append({
                'value': value,
                'timestamp': datetime.now().isoformat(),
                'step': step
            })

            # Log to MLflow
            with mlflow.start_run(run_id=run_id):
                mlflow.log_metric(key, value, step=step)

            logger.debug(f"Logged metric {key}={value} for run {run_id}")

        except Exception as e:
            logger.error(f"Error logging metric: {str(e)}")
            raise

    def log_metrics(self, run_id: str, metrics: Dict[str, float], step: int = None) -> None:
        """
        Log multiple metrics for a run.

        Args:
            run_id: Run ID
            metrics: Dictionary of metrics
            step: Step number
        """
        for key, value in metrics.items():
            self.log_metric(run_id, key, value, step)

    def log_parameter(self, run_id: str, key: str, value: Any) -> None:
        """
        Log a parameter for a run.

        Args:
            run_id: Run ID
            key: Parameter key
            value: Parameter value
        """
        try:
            run = self._find_run(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")

            run['parameters'][key] = value

            # Log to MLflow
            with mlflow.start_run(run_id=run_id):
                mlflow.log_param(key, value)

            logger.debug(f"Logged parameter {key}={value} for run {run_id}")

        except Exception as e:
            logger.error(f"Error logging parameter: {str(e)}")
            raise

    def log_artifact(self, run_id: str, local_path: str, artifact_path: str = None) -> None:
        """
        Log an artifact for a run.

        Args:
            run_id: Run ID
            local_path: Local path to the artifact
            artifact_path: Path within the artifact store
        """
        try:
            run = self._find_run(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")

            # Log to internal storage
            run['artifacts'][artifact_path or Path(local_path).name] = {
                'local_path': local_path,
                'logged_at': datetime.now().isoformat()
            }

            # Log to MLflow
            with mlflow.start_run(run_id=run_id):
                mlflow.log_artifact(local_path, artifact_path)

            logger.debug(f"Logged artifact {local_path} for run {run_id}")

        except Exception as e:
            logger.error(f"Error logging artifact: {str(e)}")
            raise

    def log_model(self, run_id: str, model, model_name: str = "model") -> None:
        """
        Log a model for a run.

        Args:
            run_id: Run ID
            model: Model object
            model_name: Name for the model
        """
        try:
            with mlflow.start_run(run_id=run_id):
                mlflow.sklearn.log_model(model, model_name)

            logger.debug(f"Logged model {model_name} for run {run_id}")

        except Exception as e:
            logger.error(f"Error logging model: {str(e)}")
            raise

    def end_run(self, run_id: str, status: str = "finished") -> None:
        """
        End a run.

        Args:
            run_id: Run ID
            status: Final status of the run
        """
        try:
            run = self._find_run(run_id)
            if not run:
                raise ValueError(f"Run {run_id} not found")

            run['status'] = status
            run['ended_at'] = datetime.now().isoformat()

            # End MLflow run
            mlflow.end_run()

            # Save metadata
            self._save_metadata()

            logger.info(f"Ended run {run_id} with status {status}")

        except Exception as e:
            logger.error(f"Error ending run: {str(e)}")
            raise

    def end_experiment(self, experiment_id: str, status: str = "completed") -> None:
        """
        End an experiment.

        Args:
            experiment_id: Experiment ID
            status: Final status of the experiment
        """
        try:
            if experiment_id not in self.experiments:
                raise ValueError(f"Experiment {experiment_id} not found")

            experiment = self.experiments[experiment_id]
            experiment['status'] = status
            experiment['ended_at'] = datetime.now().isoformat()

            # Generate summary
            experiment['summary'] = self._generate_experiment_summary(experiment_id)

            # Remove from active experiments
            if experiment_id in self.active_experiments:
                del self.active_experiments[experiment_id]

            # Save metadata
            self._save_metadata()

            logger.info(f"Ended experiment {experiment_id} with status {status}")

        except Exception as e:
            logger.error(f"Error ending experiment: {str(e)}")
            raise

    def _find_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Find a run by ID."""
        for experiment in self.experiments.values():
            for run in experiment['runs']:
                if run['run_id'] == run_id:
                    return run
        return None

    def _generate_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Generate summary statistics for an experiment."""
        try:
            experiment = self.experiments[experiment_id]
            runs = experiment['runs']

            if not runs:
                return {}

            # Collect all metrics across runs
            all_metrics = defaultdict(list)
            completed_runs = [r for r in runs if r['status'] == 'finished']

            for run in completed_runs:
                for metric_name, metric_values in run['metrics'].items():
                    if metric_values:
                        # Take the last value for each metric
                        all_metrics[metric_name].append(metric_values[-1]['value'])

            summary = {
                'total_runs': len(runs),
                'completed_runs': len(completed_runs),
                'failed_runs': len([r for r in runs if r['status'] == 'failed']),
                'metric_summaries': {}
            }

            # Calculate metric statistics
            for metric_name, values in all_metrics.items():
                if values:
                    summary['metric_summaries'][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

            return summary

        except Exception as e:
            logger.error(f"Error generating experiment summary: {str(e)}")
            return {}

    def _save_metadata(self) -> None:
        """Save experiments metadata to file."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.experiments, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving metadata: {str(e)}")
            raise

    def get_experiment(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """
        Get experiment details.

        Args:
            experiment_id: Experiment ID

        Returns:
            Experiment details
        """
        return self.experiments.get(experiment_id)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """
        Get run details.

        Args:
            run_id: Run ID

        Returns:
            Run details
        """
        return self._find_run(run_id)

    def list_experiments(self, status: str = None, tags: Dict[str, str] = None) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.

        Args:
            status: Filter by status
            tags: Filter by tags

        Returns:
            List of experiments
        """
        experiments = list(self.experiments.values())

        if status:
            experiments = [e for e in experiments if e['status'] == status]

        if tags:
            filtered = []
            for exp in experiments:
                if all(exp.get('tags', {}).get(k) == v for k, v in tags.items()):
                    filtered.append(exp)
            experiments = filtered

        return experiments

    def compare_runs(
        self,
        run_ids: List[str],
        metrics: List[str] = None
    ) -> Dict[str, Any]:
        """
        Compare multiple runs.

        Args:
            run_ids: List of run IDs to compare
            metrics: Metrics to compare

        Returns:
            Comparison results
        """
        try:
            runs_data = []
            for run_id in run_ids:
                run = self._find_run(run_id)
                if run:
                    runs_data.append(run)

            if not runs_data:
                return {}

            comparison = {
                'run_ids': run_ids,
                'runs': runs_data,
                'metric_comparison': {}
            }

            if metrics:
                for metric in metrics:
                    metric_values = []
                    for run in runs_data:
                        if metric in run['metrics'] and run['metrics'][metric]:
                            metric_values.append(run['metrics'][metric][-1]['value'])
                        else:
                            metric_values.append(None)

                    if metric_values and any(v is not None for v in metric_values):
                        valid_values = [v for v in metric_values if v is not None]
                        comparison['metric_comparison'][metric] = {
                            'values': metric_values,
                            'mean': np.mean(valid_values) if valid_values else None,
                            'std': np.std(valid_values) if valid_values else None,
                            'best_run': run_ids[np.argmax(valid_values)] if valid_values else None
                        }

            return comparison

        except Exception as e:
            logger.error(f"Error comparing runs: {str(e)}")
            raise

    def get_best_run(
        self,
        experiment_id: str,
        metric: str,
        maximize: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run for an experiment based on a metric.

        Args:
            experiment_id: Experiment ID
            metric: Metric to optimize
            maximize: Whether to maximize the metric

        Returns:
            Best run details
        """
        try:
            experiment = self.experiments.get(experiment_id)
            if not experiment:
                return None

            best_run = None
            best_value = float('-inf') if maximize else float('inf')

            for run in experiment['runs']:
                if run['status'] == 'finished' and metric in run['metrics']:
                    metric_values = run['metrics'][metric]
                    if metric_values:
                        value = metric_values[-1]['value']
                        if (maximize and value > best_value) or (not maximize and value < best_value):
                            best_value = value
                            best_run = run

            return best_run

        except Exception as e:
            logger.error(f"Error getting best run: {str(e)}")
            return None

    def export_experiment(self, experiment_id: str, output_path: str) -> None:
        """
        Export experiment data to file.

        Args:
            experiment_id: Experiment ID
            output_path: Output file path
        """
        try:
            experiment = self.experiments.get(experiment_id)
            if not experiment:
                raise ValueError(f"Experiment {experiment_id} not found")

            Path(output_path).parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, 'w') as f:
                json.dump(experiment, f, indent=2)

            logger.info(f"Exported experiment {experiment_id} to {output_path}")

        except Exception as e:
            logger.error(f"Error exporting experiment: {str(e)}")
            raise
