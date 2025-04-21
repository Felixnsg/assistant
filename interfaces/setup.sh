#!/bin/bash

# Orpheus TTS Server Setup Script

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is required but not found. Please install Python 3 and try again."
    exit 1
fi

# Check if pip is installed
if ! command -v pip &> /dev/null; then
    echo "pip is required but not found. Please install pip and try again."
    exit 1
fi

# Check for NVIDIA GPU and CUDA
echo "Checking for NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    echo "NVIDIA GPU found:"
    nvidia-smi
else
    echo "WARNING: NVIDIA GPU not detected. Orpheus TTS requires a GPU for optimal performance."
    read -p "Continue with setup anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv orpheus_env

# Activate virtual environment
echo "Activating virtual environment..."
source orpheus_env/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn pydantic

# Install Orpheus TTS
echo "Installing Orpheus TTS..."
pip install orpheus-speech

# Fix potential vllm issue
echo "Installing specific vllm version to avoid bugs..."
pip install vllm==0.7.3

echo
echo "Orpheus TTS Server setup complete!"
echo "To start the server:"
echo "1. Activate the virtual environment: source orpheus_env/bin/activate"
echo "2. Run the server: python main.py"
echo
echo "The server will be available at http://localhost:8080"
echo "API documentation is available at http://localhost:8080/docs"