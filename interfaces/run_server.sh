#!/bin/bash

echo "Starting Orpheus TTS Server..."

# Clean up GPU memory
echo "Cleaning up GPU memory..."
pkill -f python || true
sleep 2

# Configure vLLM
echo "Configuring vLLM with memory optimizations..."
python3 configure_vllm.py --max-seq-len 60000 --gpu-utilization 0.95

# Run the server
echo "Starting the server..."
python3 basic_main.py