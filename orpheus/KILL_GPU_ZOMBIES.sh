#!/bin/bash
# EMERGENCY GPU MEMORY CLEANUP SCRIPT
# Run this when you get CUDA out of memory errors

echo "==================================="
echo "KILLING ALL GPU ZOMBIE PROCESSES"
echo "==================================="

# Show current GPU usage
echo "Current GPU usage:"
nvidia-smi

echo -e "\n🔪 Killing Python processes..."
# Kill all Python processes (BE CAREFUL - this kills EVERYTHING)
pkill -f python
pkill -f orpheus
pkill -f server

# Give it a moment
sleep 2

# Force kill if still there
pkill -9 -f python
pkill -9 -f orpheus
pkill -9 -f server

echo -e "\n🧹 Clearing GPU cache..."
python3 << 'EOF'
import torch
import gc
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    print("✅ GPU cache cleared")
else:
    print("❌ No CUDA available")
EOF

echo -e "\n📊 GPU status after cleanup:"
nvidia-smi

echo -e "\n✅ Cleanup complete! You can now start the server again."
echo "Run: python3 server_async_working.py"