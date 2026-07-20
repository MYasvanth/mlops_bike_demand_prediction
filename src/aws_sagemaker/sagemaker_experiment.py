"""
AWS SageMaker Experiment Tracking
"""

import boto3
import yaml
from typing import Dict, Any, Optional, List

class SageMakerExperimentTracker:
    def __init__(self, config_path: str = "configs/aws_sagemaker_config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.config['aws']['region'])

    def start_experiment(self, experiment_name: str, trial_name: Optional[str] = None) -> str:
        self.sagemaker_client.create_experiment(
            ExperimentName=experiment_name,
            Description="Bike demand prediction experiment"
        )
        trial_name = trial_name or f"{experiment_name}-trial"
        self.sagemaker_client.create_trial(
            TrialName=trial_name,
            ExperimentName=experiment_name
        )
        return trial_name

    def log_parameters(self, parameters: Dict[str, Any], trial_name: str):
        self.sagemaker_client.create_trial_component(
            TrialComponentName=f"{trial_name}-parameters",
            Parameters={k: {'NumberValue': float(v) if isinstance(v, (int, float)) else str(v)} for k, v in parameters.items()}
        )

    def log_metrics(self, metrics: Dict[str, Any], trial_name: str):
        self.sagemaker_client.create_trial_component(
            TrialComponentName=f"{trial_name}-metrics",
            Metrics=[{
                'MetricName': k,
                'Value': float(v),
                'Timestamp': __import__('datetime').datetime.utcnow()
            } for k, v in metrics.items() if isinstance(v, (int, float))]
        )

    def list_experiments(self) -> List[Dict[str, Any]]:
        response = self.sagemaker_client.list_experiments()
        return [{'name': e['ExperimentName'], 'description': e.get('Description', '')} for e in response['ExperimentSummaryList']]

    def end_experiment(self, trial_name: str):
        self.sagemaker_client.update_trial(
            TrialName=trial_name
        )
