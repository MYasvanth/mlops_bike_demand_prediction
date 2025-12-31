from flask import Flask, request, jsonify
import logging
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
import joblib
from typing import Dict, Any, Optional
import traceback
from datetime import datetime

# Add src to path for imports
import sys
import os
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir.parent))

try:
    from src.models.bike_demand_model import BikeDemandModel
    from src.features.data_processor import DataProcessor
    from src.visualization.model_explainability import ModelExplainer
    print("Successfully imported all modules")
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path}")
    BikeDemandModel = None
    DataProcessor = None
    ModelExplainer = None

app = Flask(__name__)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables for model and processor
model = None
processor = None
explainer = None
config = None

def load_model_and_config():
    """Load the trained model, processor, and configuration."""
    global model, processor, explainer, config

    try:
        # Load configuration
        config_path = Path(__file__).parent.parent.parent / "configs" / "model_config.yaml"
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Load trained model
        model_path = Path(__file__).parent.parent.parent / "models" / "bike_demand_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Model loaded successfully")
        else:
            logger.warning("Model file not found. Please train the model first.")

        # Initialize processor
        if DataProcessor is not None:
            processor = DataProcessor(config)
            # Load transformers if they exist
            try:
                processor.load_transformers("models/transformers")
                logger.info("Transformers loaded successfully")
            except Exception as e:
                logger.warning(f"Could not load transformers: {e}")
        else:
            processor = None
            logger.warning("DataProcessor not available")

        # Initialize explainer (optional)
        try:
            explainer = ModelExplainer(config)
            logger.info("Model explainer initialized")
        except Exception as e:
            logger.warning(f"Could not initialize explainer: {e}")
            explainer = None

    except Exception as e:
        logger.error(f"Error loading model and config: {e}")
        raise

# Load model on startup
load_model_and_config()

@app.route('/', methods=['GET'])
def index():
    """Root endpoint providing API information."""
    return jsonify({
        'message': 'Bike Demand Prediction API',
        'version': config.get('model', {}).get('version', '1.0.0') if config else '1.0.0',
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /welcome': 'Welcome message',
            'POST /predict': 'Single prediction',
            'POST /predict/batch': 'Batch predictions',
            'POST /explain': 'Prediction explanation',
            'GET /model/info': 'Model information'
        },
        'documentation': 'See README.md for detailed API usage',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model is not None,
        'explainer_loaded': explainer is not None
    })

@app.route('/welcome', methods=['GET'])
def welcome():
    """
    Returns a welcome message and logs request metadata.
    """
    # Log request metadata
    logger.info(f"Request received: {request.method} {request.path}")

    # Return JSON response
    return jsonify({'message': 'Welcome to the Bike Demand Prediction API!'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict bike demand based on input features.

    Expected JSON format:
    {
        "season": 1,
        "yr": 0,
        "mnth": 1,
        "holiday": 0,
        "weekday": 1,
        "workingday": 1,
        "weathersit": 1,
        "temp": 0.25,
        "atemp": 0.3,
        "hum": 0.6,
        "windspeed": 0.2
    }
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        logger.info(f"Prediction request received: {data}")

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Validate required features
        required_features = ['season', 'yr', 'mnth', 'holiday', 'weekday',
                           'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

        missing_features = [f for f in required_features if f not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Add missing engineered features if needed
        if 'instant' not in input_df.columns:
            input_df['instant'] = 1  # Default value

        # Process input data
        processed_data = processor.preprocess_features(input_df, fit=False)
        processed_data = processor.create_features(processed_data, fit=False)

        # Add missing features that the model was trained on
        # These features are derived from the original dataset during training
        missing_features = ['casual', 'registered', 'year', 'month', 'day', 'day_of_week']
        for feature in missing_features:
            processed_data[feature] = 0  # Default values for inference

        # Ensure features are in the correct order as expected by the model
        expected_features = ['instant', 'season', 'yr', 'mnth', 'holiday', 'weekday', 'workingday',
                           'weathersit', 'temp', 'atemp', 'hum', 'windspeed', 'casual', 'registered',
                           'year', 'month', 'day', 'day_of_week', 'temp_hum_interaction', 'wind_category',
                           'temp_season_adjusted']

        # Reorder columns to match training order
        processed_data = processed_data[expected_features]

        # Make prediction
        prediction = model.predict(processed_data)

        # Prepare response
        response = {
            'prediction': float(prediction[0]),
            'input_features': data,
            'timestamp': datetime.now().isoformat(),
            'model_version': config.get('model', {}).get('version', '1.0.0') if config else 'unknown'
        }

        logger.info(f"Prediction made: {response['prediction']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Predict bike demand for multiple instances.

    Expected JSON format:
    [
        {
            "season": 1,
            "yr": 0,
            "mnth": 1,
            "holiday": 0,
            "weekday": 1,
            "workingday": 1,
            "weathersit": 1,
            "temp": 0.25,
            "atemp": 0.3,
            "hum": 0.6,
            "windspeed": 0.2
        },
        ...
    ]
    """
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500

        # Get JSON data
        data = request.get_json()
        if not data or not isinstance(data, list):
            return jsonify({'error': 'Expected a list of prediction instances'}), 400

        if len(data) > 100:
            return jsonify({'error': 'Batch size limited to 100 predictions'}), 400

        logger.info(f"Batch prediction request received: {len(data)} instances")

        # Convert to DataFrame
        input_df = pd.DataFrame(data)

        # Validate required features
        required_features = ['season', 'yr', 'mnth', 'holiday', 'weekday',
                           'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

        missing_features = [f for f in required_features if f not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Add missing engineered features if needed
        if 'instant' not in input_df.columns:
            input_df['instant'] = range(1, len(input_df) + 1)

        # Process input data
        processed_data = processor.preprocess_features(input_df, fit=False)
        processed_data = processor.create_features(processed_data, fit=False)

        # Make predictions
        predictions = model.predict(processed_data)

        # Prepare response
        response = {
            'predictions': [float(p) for p in predictions],
            'count': len(predictions),
            'timestamp': datetime.now().isoformat(),
            'model_version': config.get('model', {}).get('version', '1.0.0') if config else 'unknown'
        }

        logger.info(f"Batch predictions made: {len(predictions)} predictions")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Batch prediction failed: {str(e)}'}), 500

@app.route('/explain', methods=['POST'])
def explain_prediction():
    """
    Explain a single prediction using SHAP.

    Expected JSON format: same as /predict endpoint
    """
    try:
        if explainer is None:
            return jsonify({'error': 'Model explainer not available'}), 500

        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        # Get JSON data
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        logger.info(f"Explanation request received: {data}")

        # Convert to DataFrame
        input_df = pd.DataFrame([data])

        # Validate required features
        required_features = ['season', 'yr', 'mnth', 'holiday', 'weekday',
                           'workingday', 'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

        missing_features = [f for f in required_features if f not in input_df.columns]
        if missing_features:
            return jsonify({'error': f'Missing required features: {missing_features}'}), 400

        # Add missing engineered features
        if 'instant' not in input_df.columns:
            input_df['instant'] = 1

        # Process input data
        processed_data = processor.preprocess_data(input_df)

        # Initialize explainer with training data (if not already done)
        if not explainer.is_initialized:
            # Load some training data for explainer initialization
            try:
                train_data_path = Path("../../data/processed/train_processed.csv")
                if train_data_path.exists():
                    train_df = pd.read_csv(train_data_path).head(100)
                    train_processed = processor.preprocess_data(train_df.drop('cnt', axis=1, errors='ignore'))
                    explainer.initialize_explainer(model, train_processed)
                else:
                    return jsonify({'error': 'Training data not found for explainer initialization'}), 500
            except Exception as e:
                logger.error(f"Failed to initialize explainer: {e}")
                return jsonify({'error': f'Explainer initialization failed: {str(e)}'}), 500

        # Generate SHAP explanation
        explanation = explainer.explain_prediction_shap(processed_data)

        # Prepare response
        response = {
            'prediction': explanation['prediction'],
            'shap_values': explanation['shap_values'],
            'base_value': explanation['base_value'],
            'feature_names': explanation['feature_names'],
            'feature_values': explanation['feature_values'],
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"Explanation generated for prediction: {response['prediction']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Explanation error: {str(e)}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'Explanation failed: {str(e)}'}), 500

@app.route('/model/info', methods=['GET'])
def model_info():
    """Get information about the loaded model."""
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded'}), 500

        info = {
            'model_type': str(type(model).__name__),
            'config': config,
            'features': config.get('features', {}) if config else {},
            'target': config.get('target', 'cnt') if config else 'cnt',
            'timestamp': datetime.now().isoformat()
        }

        # Add model-specific info if available
        if hasattr(model, 'n_estimators'):
            info['n_estimators'] = model.n_estimators
        if hasattr(model, 'max_depth'):
            info['max_depth'] = model.max_depth

        return jsonify(info)

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({'error': f'Failed to get model info: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
