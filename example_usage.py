"""
Example usage of the Dish Top Model API.
This script demonstrates how to interact with the API programmatically.
"""

import requests
import json


class DishTopModelClient:
    """Client for interacting with the Dish Top Model API."""

    def __init__(self, base_url="http://localhost:5000"):
        """
        Initialize the client.

        Args:
            base_url: Base URL of the API (default: http://localhost:5000)
        """
        self.base_url = base_url.rstrip('/')

    def health_check(self):
        """
        Check if the API is healthy.

        Returns:
            dict: Health check response
        """
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_model_info(self):
        """
        Get information about the loaded model.

        Returns:
            dict: Model information
        """
        response = requests.get(f"{self.base_url}/model/info")
        response.raise_for_status()
        return response.json()

    def predict(self, features):
        """
        Make a single prediction.

        Args:
            features: List of feature values

        Returns:
            dict: Prediction response
        """
        data = {"features": features}
        response = requests.post(
            f"{self.base_url}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()

    def predict_batch(self, data):
        """
        Make batch predictions.

        Args:
            data: List of feature lists

        Returns:
            dict: Batch prediction response
        """
        payload = {"data": data}
        response = requests.post(
            f"{self.base_url}/predict/batch",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        response.raise_for_status()
        return response.json()


def main():
    """Main example function."""
    # Initialize client
    client = DishTopModelClient()

    print("=" * 60)
    print("Dish Top Model API - Example Usage")
    print("=" * 60)

    # 1. Health check
    print("\n1. Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(f"Status: {health['status']}")
        print(f"Model Loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the Flask server is running!")
        return

    # 2. Get model info
    print("\n2. Model Information")
    print("-" * 60)
    try:
        info = client.get_model_info()
        print(f"Model Type: {info.get('model_type', 'Unknown')}")
        print(f"Has Probability Predictions: {info.get('has_predict_proba', False)}")
        if 'n_features' in info:
            print(f"Number of Features: {info['n_features']}")
        if 'classes' in info:
            print(f"Classes: {info['classes']}")
    except Exception as e:
        print(f"Error: {e}")

    # 3. Single prediction
    print("\n3. Single Prediction")
    print("-" * 60)
    try:
        # Example features - adjust based on your model's requirements
        sample_features = [5.1, 3.5, 1.4, 0.2]
        print(f"Input Features: {sample_features}")

        result = client.predict(sample_features)
        print(f"Prediction: {result['prediction']}")
        if 'probabilities' in result:
            print(f"Probabilities: {result['probabilities']}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Adjust the sample_features to match your model's input requirements")

    # 4. Batch prediction
    print("\n4. Batch Prediction")
    print("-" * 60)
    try:
        # Example batch data - adjust based on your model's requirements
        batch_data = [
            [5.1, 3.5, 1.4, 0.2],
            [6.2, 3.4, 5.4, 2.3],
            [5.9, 3.0, 5.1, 1.8]
        ]
        print(f"Input Samples: {len(batch_data)}")

        result = client.predict_batch(batch_data)
        print(f"Predictions: {result['predictions']}")
        print(f"Count: {result['count']}")
        if 'probabilities' in result:
            print("Probabilities:")
            for i, probs in enumerate(result['probabilities']):
                print(f"  Sample {i+1}: {probs}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Adjust the batch_data to match your model's input requirements")

    # 5. Error handling example
    print("\n5. Error Handling Example")
    print("-" * 60)
    try:
        # This should fail validation
        invalid_features = []  # Empty features list
        result = client.predict(invalid_features)
    except requests.exceptions.HTTPError as e:
        print(f"Expected error caught:")
        error_response = e.response.json()
        print(f"Error: {error_response.get('error', 'Unknown error')}")
        if 'details' in error_response:
            print(f"Details: {json.dumps(error_response['details'], indent=2)}")

    print("\n" + "=" * 60)
    print("Example completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

