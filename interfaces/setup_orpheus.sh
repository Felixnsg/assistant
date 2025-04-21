#!/bin/bash

# Create and activate a virtual environment
echo "Creating virtual environment..."
python -m venv orpheus_env
source orpheus_env/bin/activate

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install Orpheus TTS
echo "Installing Orpheus TTS and dependencies..."
pip install orpheus-speech
pip install vllm==0.7.3  # Downgrading vllm as recommended in the docs

# Install server dependencies
echo "Installing server dependencies..."
pip install flask flask-cors

# Create the server script
echo "Creating server script..."
cat > orpheus_server.py << 'EOL'
# The server code will be pasted here during setup
EOL

echo "Installation complete! To start the server:"
echo "1. Activate the virtual environment: source orpheus_env/bin/activate"
echo "2. Run the server: python orpheus_server.py"