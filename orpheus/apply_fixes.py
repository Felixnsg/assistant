#!/usr/bin/env python3
"""
Apply critical Orpheus fixes from FIX_EVERYTHING.sh
Standalone script with no external dependencies
"""

import json
import glob
import os
from pathlib import Path

print("="*50)
print("APPLYING CRITICAL ORPHEUS FIXES")
print("="*50)

# FIX 1: HuggingFace config.json max_position_embeddings
print("\n[FIX 1] Fixing HuggingFace config.json...")

# Find config files
cache_paths = [
    Path.home() / ".cache" / "huggingface",
]

# Only add root cache if we can access it
if os.path.exists("/root") and os.access("/root", os.R_OK):
    cache_paths.append(Path("/root/.cache/huggingface"))

config_fixed = False
for cache_path in cache_paths:
    try:
        if cache_path and cache_path.exists():
            # Look for config.json files
            pattern = str(cache_path / "hub" / "models--canopylabs--orpheus-tts*" / "snapshots" / "*" / "config.json")
            configs = glob.glob(pattern)
            
            for config_file in configs:
                try:
                    print(f"Found config: {config_file}")
                    
                    with open(config_file, 'r') as f:
                        config = json.load(f)
                    
                    current_val = config.get('max_position_embeddings', 0)
                    print(f"Current max_position_embeddings: {current_val}")
                    
                    if current_val == 131072:
                        config['max_position_embeddings'] = 64000
                        
                        with open(config_file, 'w') as f:
                            json.dump(config, f, indent=2)
                        
                        print(f"✅ FIXED: max_position_embeddings 131072 -> 64000")
                        config_fixed = True
                    elif current_val == 64000:
                        print("✅ Already fixed (64000)")
                        config_fixed = True
                    else:
                        print(f"⚠️  Unexpected value: {current_val}")
                        
                except Exception as e:
                    print(f"Error processing {config_file}: {e}")
    except Exception as e:
        print(f"Error accessing cache path {cache_path}: {e}")

if not config_fixed:
    print("⚠️  Config not found in cache. It will be fixed after first model download.")
    print("    The model needs to be downloaded first before we can fix it.")

# FIX 2: UUID for duplicate request IDs in engine_class.py
print("\n[FIX 2] Fixing engine_class.py (duplicate request IDs)...")

try:
    import orpheus_tts
    engine_path = Path(orpheus_tts.__file__).parent / "engine_class.py"
    
    if engine_path.exists():
        print(f"Found engine_class.py: {engine_path}")
        
        with open(engine_path, 'r') as f:
            content = f.read()
        
        needs_fix = False
        
        # Check if uuid import is missing
        if 'import uuid' not in content:
            content = content.replace('import queue', 'import queue\nimport uuid')
            needs_fix = True
            print("Added: import uuid")
        
        # Fix the request_id parameter
        if 'request_id="req-001"' in content:
            content = content.replace('request_id="req-001"', 'request_id=None')
            
            # Add UUID generation
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'def generate_tokens_sync' in line and 'request_id=None' in line:
                    # Check if fix already applied
                    if i + 1 < len(lines) and 'if request_id is None:' not in lines[i + 1]:
                        indent = '        '
                        lines.insert(i + 1, f'{indent}if request_id is None:')
                        lines.insert(i + 2, f'{indent}    request_id = f"req-{{uuid.uuid4().hex[:8]}}"')
                        needs_fix = True
                        print("Added UUID generation for request_id")
                    break
            
            if needs_fix:
                content = '\n'.join(lines)
        
        if needs_fix:
            with open(engine_path, 'w') as f:
                f.write(content)
            print("✅ Fixed engine_class.py")
        else:
            print("✅ engine_class.py already fixed or doesn't need fixing")
            
    else:
        print("⚠️  engine_class.py not found - orpheus_tts not installed yet")
        
except ImportError:
    print("⚠️  orpheus_tts not installed yet - fix will apply after installation")
except Exception as e:
    print(f"Error: {e}")

print("\n" + "="*50)
print("FIXES APPLIED")
print("="*50)
print("\nIMPORTANT:")
print("1. If config.json wasn't found, run the model once to download it, then run this again")
print("2. The server should now work with max_tokens=64000")
print("3. Start server with: python3 orpheus_service.py")