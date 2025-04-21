#!/usr/bin/env python3

"""
This script sets up vLLM configuration for Orpheus TTS to work with limited GPU memory.
Run this script before starting the Orpheus TTS server.
"""

import os
import sys
import json
import argparse

def get_gpu_info():
    """Get GPU memory information"""
    try:
        import torch
        if not torch.cuda.is_available():
            return {"status": "No CUDA GPU available"}
        
        gpu_count = torch.cuda.device_count()
        gpus = []
        
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            total_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            free_memory, total_mem = torch.cuda.mem_get_info(i)
            free_memory = free_memory / (1024**3)
            
            gpus.append({
                "index": i,
                "name": gpu_name,
                "total_memory_gb": total_memory,
                "free_memory_gb": free_memory
            })
        
        return {
            "status": "CUDA available",
            "gpu_count": gpu_count,
            "gpus": gpus
        }
    except Exception as e:
        return {"status": f"Error: {str(e)}"}

def configure_vllm(max_seq_len=None, gpu_mem_utilization=None):
    """Configure vLLM for Orpheus TTS"""
    # Check if ~/.cache/vllm exists
    vllm_cache_dir = os.path.expanduser("~/.cache/vllm")
    if not os.path.exists(vllm_cache_dir):
        os.makedirs(vllm_cache_dir, exist_ok=True)
        print(f"Created vLLM cache directory: {vllm_cache_dir}")
    
    # Create/update configuration file
    config_file = os.path.join(vllm_cache_dir, "config.json")
    
    config = {}
    if os.path.exists(config_file):
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
        except:
            print(f"Warning: Could not read existing config at {config_file}, creating new one")
    
    # Update configuration
    if max_seq_len is not None:
        config["max_model_len"] = max_seq_len
    
    if gpu_mem_utilization is not None:
        config["gpu_memory_utilization"] = gpu_mem_utilization
    
    # Add other helpful settings
    config["tensor_parallel_size"] = 1  # Use only one GPU
    config["dtype"] = "half"  # Use half precision to save memory
    
    # Save the configuration
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"vLLM configuration saved to {config_file}:")
    print(json.dumps(config, indent=2))
    
    # Set environment variables
    os.environ["VLLM_MAX_MODEL_LEN"] = str(config.get("max_model_len", 80000))
    print(f"Set environment variable VLLM_MAX_MODEL_LEN={os.environ['VLLM_MAX_MODEL_LEN']}")
    
    # Clear any existing cache
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
    except:
        pass

def main():
    parser = argparse.ArgumentParser(description="Configure vLLM for Orpheus TTS")
    parser.add_argument("--max-seq-len", type=int, default=80000, 
                        help="Maximum sequence length (default: 80000)")
    parser.add_argument("--gpu-utilization", type=float, default=0.9,
                        help="GPU memory utilization fraction (default: 0.9)")
    parser.add_argument("--info", action="store_true", 
                        help="Show GPU information only")
    
    args = parser.parse_args()
    
    # Show GPU information
    gpu_info = get_gpu_info()
    print("GPU Information:")
    print(json.dumps(gpu_info, indent=2))
    
    if args.info:
        return
    
    # Configure vLLM
    configure_vllm(max_seq_len=args.max_seq_len, gpu_mem_utilization=args.gpu_utilization)
    
    print("\nConfiguration complete. You can now start the Orpheus TTS server.")
    print("If you still encounter memory issues, try reducing --max-seq-len further.")

if __name__ == "__main__":
    main()