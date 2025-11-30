"""
Test script for the Dish Top Model API.
Run this after starting the Flask server to test all endpoints.
"""

import requests
import json

BASE_URL = "http://localhost:5000"


def print_response(title, response):
    """Pretty print API response."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Status Code: {response.status_code}")
    print(f"Response:")
    print(json.dumps(response.json(), indent=2))


def test_health_check():
    """Test health check endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print_response("Health Check", response)
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False


def test_model_info():
    """Test model info endpoint."""
    try:
        response = requests.get(f"{BASE_URL}/model/info")
        print_response("Model Info", response)
        return response.status_code == 200
    except Exception as e:
        print(f"Model info failed: {e}")
        return False


def test_single_prediction():
    """Test single prediction endpoint."""
    try:
        # Example with 4 features - adjust based on your model
        data = {
            "features": ["12/12/2025"]
        }
        response = requests.post(
            f"{BASE_URL}/predict",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print_response("Single Prediction", response)
        return response.status_code == 200
    except Exception as e:
        print(f"Single prediction failed: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint."""
    try:
        # Example with multiple samples - adjust based on your model
        data = {
            "data": [
                [5.1, 3.5, 1.4, 0.2],
                [6.2, 3.4, 5.4, 2.3],
                [5.9, 3.0, 5.1, 1.8]
            ]
        }
        response = requests.post(
            f"{BASE_URL}/predict/batch",
            json=data,
            headers={"Content-Type": "application/json"}
        )
        print_response("Batch Prediction", response)
        return response.status_code == 200
    except Exception as e:
        print(f"Batch prediction failed: {e}")
        return False


def test_validation_errors():
    """Test validation error handling."""
    print(f"\n{'='*60}")
    print("Testing Validation Errors")
    print(f"{'='*60}")

    # Test empty features
    try:
        data = {"features": []}
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print("\nTest 1: Empty features list")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Test 1 failed: {e}")

    # Test missing features key
    try:
        data = {}
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print("\nTest 2: Missing features key")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Test 2 failed: {e}")

    # Test non-numeric features
    try:
        data = {"features": [1.0, "invalid", 3.0]}
        response = requests.post(f"{BASE_URL}/predict", json=data)
        print("\nTest 3: Non-numeric features")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Test 3 failed: {e}")


def test_404_endpoint():
    """Test 404 error handling."""
    try:
        response = requests.get(f"{BASE_URL}/nonexistent")
        print_response("404 Test", response)
        return response.status_code == 404
    except Exception as e:
        print(f"404 test failed: {e}")
        return False


def run_all_tests():
    """Run all API tests."""
    print("\n" + "="*60)
    print("Dish Top Model API Test Suite")
    print("="*60)
    print(f"\nTesting API at: {BASE_URL}")
    print("Make sure the Flask server is running before running these tests!")
    print("\nStarting tests...\n")

    tests = [
        ("Health Check", test_health_check),
        ("Model Info", test_model_info),
        ("Single Prediction", test_single_prediction),
        ("Batch Prediction", test_batch_prediction),
        ("404 Handling", test_404_endpoint),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\nTest '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Run validation tests separately (they're expected to fail with 400)
    test_validation_errors()

    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{test_name}: {status}")

    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    test_single_prediction()

