#!/bin/bash

# Startup script for Dish Top Model API

echo "========================================"
echo "Dish Top Model API - Startup Script"
echo "========================================"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate
echo "✓ Virtual environment activated"

# Install/update dependencies
echo ""
echo "Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "✓ Dependencies installed"

# Check if model file exists
if [ ! -f "model/dish_top_model.joblib" ]; then
    echo ""
    echo "⚠️  WARNING: Model file not found at model/dish_top_model.joblib"
    echo "   Please ensure your trained model is in the correct location."
    echo ""
fi

# Start the Flask application
echo ""
echo "========================================"
echo "Starting Flask server..."
echo "========================================"
echo "Server will be available at: http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python app.py

