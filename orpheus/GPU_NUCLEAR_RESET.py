#!/usr/bin/env python3
"""
NUCLEAR GPU RESET - When nothing else works
"""

import torch
import gc
import os
import sys

print("🔥 NUCLEAR GPU MEMORY RESET 🔥")
print("=" * 40)

# Method 1: Force clear everything
try:
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory before: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB allocated")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f}GB reserved")
        
        # Clear everything
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # Force garbage collection
        for _ in range(3):
            gc.collect()
            torch.cuda.empty_cache()
        
        print(f"Memory after: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB allocated")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1e9:.2f}GB reserved")
        
    else:
        print("No CUDA available")
except Exception as e:
    print(f"Error: {e}")

# Method 2: Reset the CUDA context (more aggressive)
try:
    print("\nTrying CUDA context reset...")
    import torch.cuda
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.reset_accumulated_memory_stats()
    print("✅ CUDA stats reset")
except Exception as e:
    print(f"Could not reset CUDA stats: {e}")

# Method 3: Set memory fraction
try:
    print("\nSetting memory fraction to limit usage...")
    torch.cuda.set_per_process_memory_fraction(0.9)  # Use only 90% of GPU
    print("✅ Memory fraction set to 90%")
except Exception as e:
    print(f"Could not set memory fraction: {e}")

print("\n" + "=" * 40)
print("If this doesn't work, you need to:")
print("1. Try: sudo nvidia-smi --gpu-reset")
print("2. Or restart the container/instance")
print("3. Or use a smaller model")
print("\nTo start with reduced memory:")
print("MAX_MODEL_LEN=32000 python3 server_async_working.py")