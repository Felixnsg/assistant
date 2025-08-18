#!/bin/bash

# Orpheus Enhanced TTS Server Startup Script
# Professional production deployment

echo "=========================================="
echo "🚀 ORPHEUS ENHANCED TTS SERVER"
echo "=========================================="

# Set environment variables for enhanced performance
export MODEL_NAME="canopylabs/orpheus-tts-0.1-finetune-prod"
export MAX_MODEL_LEN="64000"
export GPU_MEMORY_UTILIZATION="0.95"
export MAX_TOKENS_PER_CHUNK="8000"
export ENABLE_CHUNKING="true"
export CHUNK_OVERLAP_TOKENS="100"

# CUDA optimizations
export CUDA_VISIBLE_DEVICES="0"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# vLLM optimizations
export VLLM_USE_MODELSCOPE="false"
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"

# Logging
export LOG_LEVEL="INFO"

echo "📋 Configuration:"
echo "  - Model: $MODEL_NAME"
echo "  - Max tokens: $MAX_MODEL_LEN"
echo "  - Chunking: $ENABLE_CHUNKING"
echo "  - GPU memory: $GPU_MEMORY_UTILIZATION"
echo ""

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "🎮 GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
    echo ""
fi

# Kill any existing Orpheus processes
echo "🔄 Checking for existing processes..."
pkill -f "orpheus_enhanced_server.py" 2>/dev/null
pkill -f "server_async_working.py" 2>/dev/null
sleep 2

# Start the enhanced server
echo "🚀 Starting Enhanced Orpheus Server..."
echo "=========================================="
echo ""

# Run with optimized Python settings
python3 -u orpheus_enhanced_server.py