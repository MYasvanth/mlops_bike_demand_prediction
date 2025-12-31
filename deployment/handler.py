import json
import joblib
import pandas as pd
import numpy as np
import os
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and transformers
model = None
transformers = {}

def load_model():
    """Load model and transformers from S3 or local path"""
    global model, transformers

    if model is None:
        try:
            # In serverless environment, model path might be different
            model_path = os.environ.get('MODEL_PATH', 'models/bike_demand_model.joblib')

            if model_path.startswith('s3://'):
                # Load from S3 (would need boto3)
                import boto3
                s3 = boto3.client('s3')
                bucket, key = model_path.replace('s3://', '').split('/', 1)
                # Download and load model
                pass
            else:
                # Load from local file
                model = joblib.load(model_path)
                logger.info("Model loaded successfully")

                # Load transformers
                transformers_path = Path(model_path).parent / "transformers"
                if transformers_path.exists():
                    for file in transformers_path.glob("*.joblib"):
                        name = file.stem
                        transformers[name] = joblib.load(file)
                    logger.info(f"Loaded {len(transformers)} transformers")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

def predict(event, context):
    """Serverless prediction function"""
    try:
        load_model()

        # Parse input
        body = json.loads(event['body'])
        data = pd.DataFrame([body])

        # Apply transformations
        for name, transformer in transformers.items():
            if 'scaler' in name:
                col = name.replace('scaler_', '')
                if col in data.columns:
                    data[col] = transformer.transform(data[[col]])
            elif 'encoder' in name:
                col = name.replace('encoder_', '')
                if col in data.columns:
                    data[col] = transformer.transform(data[[col]])

        # Make prediction
        prediction = model.predict(data)[0]

        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': float(prediction),
                'features_used': list(data.columns)
            })
        }

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': 'Prediction failed',
                'message': str(e)
            })
        }

def health(event, context):
    """Health check function"""
    try:
        load_model()
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'healthy',
                'model_loaded': model is not None,
                'transformers_loaded': len(transformers)
            })
        }
    except Exception as e:
        return {
            'statusCode': 503,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'status': 'unhealthy',
                'error': str(e)
            })
        }
