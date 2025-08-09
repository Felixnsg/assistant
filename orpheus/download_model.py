#!/usr/bin/env python3
"""
Download Orpheus model files for offline use
"""

import os
import sys
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

def download_orpheus_model():
    """Download model files to local cache"""
    
    model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
    cache_dir = "/root/.cache/huggingface/hub"
    
    print(f"Downloading {model_name}...")
    print(f"Cache directory: {cache_dir}")
    
    try:
        # Method 1: Using huggingface_hub
        print("\nAttempting snapshot download...")
        local_path = snapshot_download(
            repo_id=model_name,
            cache_dir=cache_dir,
            resume_download=True,
            local_files_only=False,
            token=os.getenv("HF_TOKEN")
        )
        print(f"✅ Model downloaded to: {local_path}")
        
        # Also download tokenizer if different
        tokenizer_name = "canopylabs/orpheus-3b-0.1-pretrained"
        print(f"\nDownloading tokenizer: {tokenizer_name}")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            cache_dir=cache_dir,
            token=os.getenv("HF_TOKEN")
        )
        print("✅ Tokenizer downloaded")
        
        return local_path
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        print("\nTroubleshooting steps:")
        print("1. Check internet connection: curl -I https://huggingface.co")
        print("2. Verify token: echo $HF_TOKEN")
        print("3. Try with wget/curl directly:")
        print(f"   wget https://huggingface.co/{model_name}/resolve/main/config.json")
        return None

def test_connectivity():
    """Test network connectivity"""
    import subprocess
    
    print("Testing connectivity...")
    
    # Test basic internet
    result = subprocess.run(["curl", "-I", "https://google.com"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ Internet connection OK")
    else:
        print("❌ No internet connection")
        
    # Test HuggingFace
    result = subprocess.run(["curl", "-I", "https://huggingface.co"], 
                          capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ HuggingFace accessible")
    else:
        print("❌ Cannot reach HuggingFace")
        print("Output:", result.stderr)
        
    # Check for proxy
    proxy = os.getenv("HTTP_PROXY") or os.getenv("HTTPS_PROXY")
    if proxy:
        print(f"ℹ️ Proxy detected: {proxy}")
    
    # Check DNS
    result = subprocess.run(["nslookup", "huggingface.co"], 
                          capture_output=True, text=True)
    print(f"DNS resolution: {result.stdout[:200]}")

if __name__ == "__main__":
    print("=" * 60)
    print("ORPHEUS MODEL DOWNLOADER")
    print("=" * 60)
    
    # Test connectivity first
    test_connectivity()
    
    print("\n" + "=" * 60)
    
    # Check token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    if not token:
        print("⚠️ No HuggingFace token found in environment")
        print("Run: export HF_TOKEN='your_token_here'")
        sys.exit(1)
    else:
        print(f"✅ Token found: {token[:10]}...")
    
    # Try to download
    if "--download" in sys.argv:
        download_orpheus_model()
    else:
        print("\nRun with --download to attempt model download")
        print("Or use manual download instructions above")