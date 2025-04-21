#!/bin/bash

echo "========================================================"
echo "         Cleaning up GPU memory for Orpheus TTS         "
echo "========================================================"

# List processes using GPU
echo "Current processes using GPU:"
nvidia-smi

# Kill Python processes
echo "Killing Python processes using GPU..."
pkill -f python

# Kill any process that might be using GPU memory
# WARNING: This is more aggressive - only uncomment if the above doesn't work
# nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9

# Wait a moment
sleep 2

# Check again
echo "GPU memory after cleanup:"
nvidia-smi

# Set memory allocation config for PyTorch
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run the server with memory configuration
echo "Starting Orpheus TTS server with memory optimizations..."
CUDA_VISIBLE_DEVICES=0 python patched_main.py