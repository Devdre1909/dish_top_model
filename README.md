# Dish Top Model API

A Flask-based REST API for serving predictions from the Dish Top Model.

## Features

- 🚀 Single and batch prediction endpoints
- ✅ Request validation using Pydantic
- 🔒 CORS enabled
- 📊 Model information endpoint
- 💚 Health check endpoint
- 📝 Comprehensive error handling
- 🐳 Production-ready with Gunicorn

## Installation

### Prerequisites

- Python 3.8 or higher
- pip

### Setup

1. Clone the repository (if applicable):
```bash
cd /Users/temitope/Documents/projects/dish_top_model
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Development Mode

Run the Flask development server:

```bash
python app.py
```

The API will be available at `http://localhost:5000`

### Production Mode

Use Gunicorn for production deployment:

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

Options:
- `-w 4`: Use 4 worker processes
- `-b 0.0.0.0:5000`: Bind to all interfaces on port 5000

### Environment Variables

- `PORT`: Port to run the server (default: 5000)
- `DEBUG`: Enable debug mode (default: False)

Example:
```bash
PORT=8000 DEBUG=true python app.py
```

## API Endpoints

### Health Check

Check if the API and model are loaded successfully.

**Endpoint:** `GET /health`

**Response:**
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Single Prediction

Make a prediction for a single sample.

**Endpoint:** `POST /predict`

**Request Body:**
```json
{
  "features": [1.5, 2.3, 4.5, 0.8]
}
```

**Response:**
```json
{
  "prediction": 1,
  "probabilities": [0.23, 0.77],
  "success": true
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1.5, 2.3, 4.5, 0.8]}'
```

### Batch Prediction

Make predictions for multiple samples at once.

**Endpoint:** `POST /predict/batch`

**Request Body:**
```json
{
  "data": [
    [1.5, 2.3, 4.5, 0.8],
    [2.1, 3.2, 5.1, 1.2],
    [0.9, 1.8, 3.2, 0.5]
  ]
}
```

**Response:**
```json
{
  "predictions": [1, 1, 0],
  "probabilities": [
    [0.23, 0.77],
    [0.15, 0.85],
    [0.68, 0.32]
  ],
  "count": 3,
  "success": true
}
```

**cURL Example:**
```bash
curl -X POST http://localhost:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"data": [[1.5, 2.3, 4.5, 0.8], [2.1, 3.2, 5.1, 1.2]]}'
```

### Model Information

Get information about the loaded model.

**Endpoint:** `GET /model/info`

**Response:**
```json
{
  "model_type": "RandomForestClassifier",
  "has_predict_proba": true,
  "has_feature_importances": true,
  "n_features": 4,
  "classes": [0, 1],
  "success": true
}
```

**cURL Example:**
```bash
curl http://localhost:5000/model/info
```

## Error Handling

All endpoints return appropriate HTTP status codes and error messages:

- `200`: Success
- `400`: Bad Request (validation error or missing data)
- `404`: Endpoint not found
- `405`: Method not allowed
- `500`: Internal server error
- `503`: Service unavailable (model not loaded)

**Error Response Format:**
```json
{
  "error": "Error message",
  "details": {},
  "success": false
}
```

## Request Validation

The API uses Pydantic for request validation:

- **Features must be numeric**: All feature values must be numbers (int or float)
- **Non-empty arrays**: Feature lists and data arrays cannot be empty
- **Proper structure**: Requests must follow the specified JSON structure

## Testing

### Test with Python

```python
import requests

# Health check
response = requests.get('http://localhost:5000/health')
print(response.json())

# Single prediction
data = {
    "features": [1.5, 2.3, 4.5, 0.8]
}
response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())

# Batch prediction
data = {
    "data": [
        [1.5, 2.3, 4.5, 0.8],
        [2.1, 3.2, 5.1, 1.2]
    ]
}
response = requests.post('http://localhost:5000/predict/batch', json=data)
print(response.json())
```

## Model File

The API expects the trained model to be located at:
```
model/dish_top_model.joblib
```

The model should be trained and saved using scikit-learn's `joblib.dump()` function.

## Deployment

### Docker (Optional)

You can containerize the application for easier deployment:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

Build and run:
```bash
docker build -t dish-top-model-api .
docker run -p 5000:5000 dish-top-model-api
```

## License

MIT License

## Contributing

Feel free to submit issues and enhancement requests!

