import json
import os
import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

def load_model_and_processors():
    """Load the trained model and data processors"""
    try:
        # Load model
        model_path = project_root / "models" / "bike_demand_model.joblib"
        model = joblib.load(model_path)

        # Load data processor
        from src.features.data_processor import DataProcessor
        from src.api.fastapi_app import config

        processor = DataProcessor(config)
        transformers_path = project_root / "models" / "transformers"
        processor.load_transformers(str(transformers_path))

        return model, processor
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

# Load model and processor at startup
model, processor = load_model_and_processors()

def handler(event, context):
    """Netlify function handler"""

    # Handle CORS preflight requests
    if event['httpMethod'] == 'OPTIONS':
        return {
            'statusCode': 200,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS'
            },
            'body': ''
        }

    try:
        # Health check endpoint
        if event['path'] == '/api/health' and event['httpMethod'] == 'GET':
            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'status': 'healthy',
                    'timestamp': datetime.now().isoformat(),
                    'model_loaded': model is not None,
                    'processor_loaded': processor is not None,
                    'version': '1.0.0'
                })
            }

        # Prediction endpoint
        elif event['path'] == '/api/predict' and event['httpMethod'] == 'POST':
            if model is None or processor is None:
                return {
                    'statusCode': 503,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'error': 'Model or processor not loaded',
                        'timestamp': datetime.now().isoformat()
                    })
                }

            # Parse request body
            try:
                body = json.loads(event['body'])
            except:
                return {
                    'statusCode': 400,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'error': 'Invalid JSON in request body',
                        'timestamp': datetime.now().isoformat()
                    })
                }

            start_time = datetime.now()

            try:
                # Convert to DataFrame
                input_df = pd.DataFrame([body])
                input_df['instant'] = 1  # Default value

                # Process input data
                processed_data = processor.preprocess_features(input_df, fit=False)
                processed_data = processor.create_features(processed_data, fit=False)

                # Add missing features
                missing_features = ['casual', 'registered', 'year', 'month', 'day', 'day_of_week']
                for feature in missing_features:
                    processed_data[feature] = 0

                # Ensure features are in the correct order
                expected_features = ['instant', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                                   'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered',
                                   'year', 'month', 'day', 'day_of_week', 'temp_hum_interaction', 'wind_category',
                                   'temp_season_adjusted']

                processed_data = processed_data[expected_features]

                # Make prediction
                prediction = model.predict(processed_data)

                # Calculate processing time
                processing_time = (datetime.now() - start_time).total_seconds() * 1000

                response = {
                    'prediction': float(prediction[0]),
                    'input_features': body,
                    'model_version': '1.0.0',
                    'timestamp': datetime.now().isoformat(),
                    'processing_time_ms': processing_time
                }

                return {
                    'statusCode': 200,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps(response)
                }

            except Exception as e:
                return {
                    'statusCode': 500,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'error': f'Prediction failed: {str(e)}',
                        'timestamp': datetime.now().isoformat()
                    })
                }

        # Model info endpoint
        elif event['path'] == '/api/model/info' and event['httpMethod'] == 'GET':
            if model is None:
                return {
                    'statusCode': 503,
                    'headers': {
                        'Access-Control-Allow-Origin': '*',
                        'Content-Type': 'application/json'
                    },
                    'body': json.dumps({
                        'error': 'Model not loaded',
                        'timestamp': datetime.now().isoformat()
                    })
                }

            info = {
                'model_type': str(type(model).__name__),
                'features': ['season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                           'weathersit', 'temp', 'atemp', 'hum', 'windspeed'],
                'target': 'cnt',
                'timestamp': datetime.now().isoformat(),
                'api_version': '1.0.0'
            }

            # Add model-specific info if available
            if hasattr(model, 'n_estimators'):
                info['n_estimators'] = model.n_estimators
            if hasattr(model, 'max_depth'):
                info['max_depth'] = model.max_depth

            return {
                'statusCode': 200,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps(info)
            }

        # Method not allowed
        else:
            return {
                'statusCode': 405,
                'headers': {
                    'Access-Control-Allow-Origin': '*',
                    'Content-Type': 'application/json'
                },
                'body': json.dumps({
                    'error': 'Method not allowed',
                    'timestamp': datetime.now().isoformat()
                })
            }

    except Exception as e:
        return {
            'statusCode': 500,
            'headers': {
                'Access-Control-Allow-Origin': '*',
                'Content-Type': 'application/json'
            },
            'body': json.dumps({
                'error': f'Internal server error: {str(e)}',
                'timestamp': datetime.now().isoformat()
            })
        }
