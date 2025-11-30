from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
from pydantic import BaseModel, ValidationError, field_validator
from typing import List, Optional
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'dish_top_model.joblib')
model = None


def load_model():
    """Load the trained model from disk."""
    global model
    try:
        model = joblib.load(MODEL_PATH)
        logger.info(f"Model loaded successfully from {MODEL_PATH}")
        return True
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        return False


# Pydantic models for request validation
class PredictionRequest(BaseModel):
    """Request model for prediction endpoint."""
    features: List[float]

    @field_validator('features')
    @classmethod
    def validate_features(cls, v):
        if not v:
            raise ValueError('Features list cannot be empty')
        if any(not isinstance(x, (int, float)) for x in v):
            raise ValueError('All features must be numeric values')
        return v


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction endpoint."""
    data: List[List[float]]

    @field_validator('data')
    @classmethod
    def validate_data(cls, v):
        if not v:
            raise ValueError('Data list cannot be empty')
        if len(v) == 0:
            raise ValueError('At least one sample is required')
        return v


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    }), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Prediction endpoint for single sample.

    Expects JSON body with:
    {
        "features": [feature1, feature2, ...]
    }

    Returns:
    {
        "prediction": prediction_result,
        "success": true
    }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 503

    try:
        # Parse and validate request
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'success': False
            }), 400

        # Validate using Pydantic
        prediction_request = PredictionRequest(**data)

        # Convert to numpy array and reshape for prediction
        features = np.array(prediction_request.features).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)

        # Check if model has predict_proba method for probabilities
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features).tolist()

        response = {
            'prediction': prediction.tolist()[0] if isinstance(prediction, np.ndarray) else prediction,
            'success': True
        }

        if probabilities:
            response['probabilities'] = probabilities[0]

        return jsonify(response), 200

    except ValidationError as e:
        return jsonify({
            'error': 'Validation error',
            'details': e.errors(),
            'success': False
        }), 400
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({
            'error': f'Prediction failed: {str(e)}',
            'success': False
        }), 500


@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint for multiple samples.

    Expects JSON body with:
    {
        "data": [
            [feature1, feature2, ...],
            [feature1, feature2, ...],
            ...
        ]
    }

    Returns:
    {
        "predictions": [prediction1, prediction2, ...],
        "count": number_of_predictions,
        "success": true
    }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 503

    try:
        # Parse and validate request
        data = request.get_json()
        if not data:
            return jsonify({
                'error': 'No JSON data provided',
                'success': False
            }), 400

        # Validate using Pydantic
        batch_request = BatchPredictionRequest(**data)

        # Convert to numpy array
        features = np.array(batch_request.data)

        # Make predictions
        predictions = model.predict(features)

        # Check if model has predict_proba method for probabilities
        probabilities = None
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(features).tolist()

        response = {
            'predictions': predictions.tolist() if isinstance(predictions, np.ndarray) else predictions,
            'count': len(predictions),
            'success': True
        }

        if probabilities:
            response['probabilities'] = probabilities

        return jsonify(response), 200

    except ValidationError as e:
        return jsonify({
            'error': 'Validation error',
            'details': e.errors(),
            'success': False
        }), 400
    except Exception as e:
        logger.error(f"Batch prediction error: {str(e)}")
        return jsonify({
            'error': f'Batch prediction failed: {str(e)}',
            'success': False
        }), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model.

    Returns:
    {
        "model_type": model_class_name,
        "has_predict_proba": boolean,
        "success": true
    }
    """
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'success': False
        }), 503

    try:
        info = {
            'model_type': type(model).__name__,
            'has_predict_proba': hasattr(model, 'predict_proba'),
            'success': True
        }

        # Try to get additional model attributes if available
        if hasattr(model, 'feature_importances_'):
            info['has_feature_importances'] = True

        if hasattr(model, 'n_features_in_'):
            info['n_features'] = model.n_features_in_

        if hasattr(model, 'classes_'):
            info['classes'] = model.classes_.tolist()

        return jsonify(info), 200

    except Exception as e:
        logger.error(f"Model info error: {str(e)}")
        return jsonify({
            'error': f'Failed to get model info: {str(e)}',
            'success': False
        }), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({
        'error': 'Endpoint not found',
        'success': False
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):
    """Handle 405 errors."""
    return jsonify({
        'error': 'Method not allowed',
        'success': False
    }), 405


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal error: {str(error)}")
    return jsonify({
        'error': 'Internal server error',
        'success': False
    }), 500


if __name__ == '__main__':
    # Load model before starting the server
    if not load_model():
        logger.error("Failed to load model. Server starting but predictions will fail.")

    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'False').lower() == 'true'

    logger.info(f"Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)

