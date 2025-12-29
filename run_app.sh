#!/bin/bash

echo "======================================"
echo "AI Dermatology Classifier Setup"
echo "======================================"
echo ""

# Check Python version
echo "Checking Python version..."
python --version

# Install requirements
echo ""
echo "Installing requirements..."
pip install -q -r requirements.txt

# Check model file
echo ""
echo "Checking model file..."
if [ -f "efficientnet_best.pth" ]; then
    echo "✓ Model file found: efficientnet_best.pth"
    ls -lh efficientnet_best.pth
else
    echo "✗ Model file not found: efficientnet_best.pth"
    echo "Please ensure the model file is in the current directory"
    exit 1
fi

# Launch app
echo ""
echo "======================================"
echo "Launching Streamlit app..."
echo "======================================"
echo ""
streamlit run dermatology_app.py
