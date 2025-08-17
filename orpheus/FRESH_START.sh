#!/bin/bash
# FRESH START - When you just want it to WORK again

echo "🆕 FRESH START - Starting from scratch"
echo "====================================="

# Step 1: Nuclear option - restart Python
echo "1. Killing everything..."
pkill -9 -f python
sleep 2

# Step 2: Clear GPU if possible
echo "2. Clearing GPU..."
python3 << 'EOF'
import torch, gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print("   GPU cleared")
EOF

# Step 3: Start with REDUCED settings that WILL work
echo "3. Starting server with SAFE settings..."
echo "   (Reduced to 32000 tokens, about 6 minutes audio)"

# Start with conservative settings
export MAX_MODEL_LEN=32000
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Use the working server
echo "Starting server_async_working.py with reduced memory..."
python3 server_async_working.py