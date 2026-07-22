"""
AWS SageMaker Pipeline Orchestration
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import boto3
    from sagemaker.workflow.pipeline import Pipeline
    SAGEMAKER_AVAILABLE = True
except ImportError:
    SAGEMAKER_AVAILABLE = False

from ..utils import setup_logging

logger = setup_logging(__name__)

class SageMakerPipeline:
    def __init__(self, config_path: str = "configs/aws_sagemaker_config.yaml"):
        if not SAGEMAKER_AVAILABLE:
            raise ImportError("AWS SageMaker SDK required.")
        
        self.config = self._load_config(config_path)
        self.sagemaker_client = boto3.client('sagemaker', region_name=self.config['aws']['region'])

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, 'r') as f:
            return yaml.safe_load(f)

    def run_pipeline(self, input_data_uri: str, output_data_uri: Optional[str] = None, experiment_name: Optional[str] = None) -> Dict[str, Any]:
        pipeline_name = f"{self.config['model_registry']['domain_name']}-pipeline"
        
        parameters = [
            {'Name': 'InputDataPath', 'Value': input_data_uri}
        ]
        if output_data_uri:
            parameters.append({'Name': 'ModelOutputPath', 'Value': output_data_uri})

        response = self.sagemaker_client.start_pipeline_execution(
            PipelineName=pipeline_name,
            PipelineParameters=parameters
        )
        return {
            'job_name': response['PipelineExecutionArn'].split('/')[-1],
            'status': 'Starting'
        }

    def wait_for_pipeline_completion(self, execution_arn: str) -> str:
        import time
        while True:
            response = self.sagemaker_client.describe_pipeline_execution(
                PipelineExecutionArn=execution_arn
            )
            status = response['PipelineExecutionStatus']
            if status in ['Succeeded', 'Failed', 'Stopped']:
                return status
            time.sleep(30)
