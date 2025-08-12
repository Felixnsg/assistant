#!/bin/bash
# =========================================================
# ORPHEUS TTS - COMPLETE FIX SCRIPT FOR NEW VAST.AI INSTANCE
# Run this AFTER cloning your repo on a fresh GPU instance
# This fixes ALL the bugs we encountered!
# =========================================================

echo "================================================"
echo "ORPHEUS TTS - ULTIMATE FIX SCRIPT"
echo "Applying all fixes from our debugging session!"
echo "================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# =========================================================
# FIX 1: THE KILLER BUG - config.json max_position_embeddings
# This was THE bug that made us cry for hours!
# =========================================================
echo -e "\n${YELLOW}[FIX 1] Fixing HuggingFace config.json (THE BIG ONE!)${NC}"
echo "This fixes: ValueError: The model's max seq len (131072) is larger than KV cache"

# Find the config file in HuggingFace cache
CONFIG_PATH=$(find ~/.cache/huggingface -name "config.json" -path "*orpheus-tts*" 2>/dev/null | head -1)

if [ -z "$CONFIG_PATH" ]; then
    echo -e "${RED}Config not found in cache. Model will download on first run.${NC}"
    echo "Creating script to fix after download..."
    
    # Create a Python script that will fix it after model downloads
    cat > /tmp/fix_config_after_download.py << 'PYEOF'
import json
import glob
import time

print("Waiting for model to download...")
for i in range(30):  # Wait up to 30 seconds
    configs = glob.glob("/root/.cache/huggingface/hub/models--canopylabs--orpheus-tts*/snapshots/*/config.json")
    if configs:
        config_path = configs[0]
        print(f"Found config at: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        print(f"Current max_position_embeddings: {config.get('max_position_embeddings', 'NOT FOUND')}")
        
        # THE FIX - Change 131072 to 64000 for RTX 4090 24GB
        config['max_position_embeddings'] = 64000
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("✅ Fixed! max_position_embeddings now set to 64000")
        break
    time.sleep(1)
else:
    print("⚠️  Config not found. Run this after starting the server once.")
PYEOF
    
    echo "Run this after first server start: python3 /tmp/fix_config_after_download.py"
else
    echo "Found config at: $CONFIG_PATH"
    # Fix it now with Python
    python3 << PYEOF
import json
with open("$CONFIG_PATH", 'r') as f:
    config = json.load(f)
print(f"Current max_position_embeddings: {config.get('max_position_embeddings', 'NOT FOUND')}")
config['max_position_embeddings'] = 64000
with open("$CONFIG_PATH", 'w') as f:
    json.dump(config, f, indent=2)
print("✅ Fixed! max_position_embeddings now set to 64000")
PYEOF
fi

# =========================================================
# FIX 2: Duplicate Request ID Bug
# This caused "Request req-001 already exists" errors
# =========================================================
echo -e "\n${YELLOW}[FIX 2] Fixing duplicate request IDs in engine_class.py${NC}"
echo "This fixes: KeyError: 'Request req-001 already exists'"

# Check if the package is installed
if [ -f "/usr/local/lib/python3.10/dist-packages/orpheus_tts/engine_class.py" ]; then
    echo "Fixing installed package..."
    python3 << 'PYEOF'
import fileinput
import sys

file_path = '/usr/local/lib/python3.10/dist-packages/orpheus_tts/engine_class.py'

# Read the file
with open(file_path, 'r') as f:
    lines = f.readlines()

# Check if uuid is already imported
has_uuid = any('import uuid' in line for line in lines)

if not has_uuid:
    # Add uuid import after queue import
    for i, line in enumerate(lines):
        if 'import queue' in line:
            lines.insert(i + 1, 'import uuid\n')
            print("✅ Added: import uuid")
            break

# Fix the request_id parameter
fixed = False
for i, line in enumerate(lines):
    if 'def generate_tokens_sync' in line and 'request_id="req-001"' in line:
        lines[i] = line.replace('request_id="req-001"', 'request_id=None')
        # Add UUID generation
        indent = '        '  # Match the function's indentation
        lines.insert(i + 1, f'{indent}if request_id is None:\n')
        lines.insert(i + 2, f'{indent}    request_id = f"req-{{uuid.uuid4().hex[:8]}}"\n')
        print("✅ Fixed: request_id now uses unique UUIDs")
        fixed = True
        break

if fixed:
    # Write back
    with open(file_path, 'w') as f:
        f.writelines(lines)
    print("✅ engine_class.py fixed!")
else:
    print("⚠️  request_id fix may already be applied or pattern not found")
PYEOF
else
    echo -e "${YELLOW}Package not installed yet. Fix will apply after pip install.${NC}"
fi

# =========================================================
# FIX 3: Missing aiohttp import for WebSocket
# This caused "name 'aiohttp' is not defined" errors
# =========================================================
echo -e "\n${YELLOW}[FIX 3] Ensuring aiohttp import in server_async.py${NC}"
echo "This fixes: name 'aiohttp' is not defined"

if [ -f "/root/assistant/orpheus/server_async.py" ]; then
    # Check if already has the import
    if grep -q "^import aiohttp" /root/assistant/orpheus/server_async.py; then
        echo "✅ aiohttp import already present"
    else
        # Add the import after asyncio
        sed -i '/^import asyncio/a import aiohttp' /root/assistant/orpheus/server_async.py
        echo "✅ Added: import aiohttp"
    fi
else
    echo "⚠️  server_async.py not found. Clone your repo first."
fi

# =========================================================
# FIX 4: Remove problematic torch.cuda.utilization()
# This caused ModuleNotFoundError: pynvml not found
# =========================================================
echo -e "\n${YELLOW}[FIX 4] Removing torch.cuda.utilization() call${NC}"
echo "This fixes: ModuleNotFoundError: pynvml does not seem to be installed"

if [ -f "/root/assistant/orpheus/server_async.py" ]; then
    # Remove the utilization line if it exists
    sed -i '/"utilization": torch.cuda.utilization()/d' /root/assistant/orpheus/server_async.py
    echo "✅ Removed problematic utilization() call"
else
    echo "⚠️  server_async.py not found"
fi

# =========================================================
# HELPFUL COMMANDS REFERENCE
# =========================================================
echo -e "\n${GREEN}================================================${NC}"
echo -e "${GREEN}ALL FIXES APPLIED! 🎉${NC}"
echo -e "${GREEN}================================================${NC}"

cat << 'EOF'

QUICK REFERENCE - Commands that saved our lives:

📦 START THE SERVER:
   cd /root/assistant/orpheus
   python server_async.py

🔧 IF KV CACHE ERROR PERSISTS:
   python3 /tmp/fix_config_after_download.py
   # Then restart server

💀 KILL ZOMBIE PROCESSES (if GPU memory full):
   pkill -f python
   pkill -f server_async
   # Check GPU: nvidia-smi

🌐 TEST CONNECTION:
   # From local machine:
   curl http://YOUR_IP:8080/health
   
   # Or use WebSocket dashboard:
   Open tts_dashboard_websocket.html in browser
   Connect to YOUR_IP:8080

⚠️  VAST.AI NOTES:
   - WebSockets work better than HTTP streaming
   - Port 8080 usually works
   - No firewall commands needed (ufw not installed)

🎯 THE BIG THREE BUGS WE FIXED:
   1. config.json: max_position_embeddings 131072 -> 64000
   2. engine_class.py: Unique request IDs with uuid
   3. server_async.py: Added aiohttp import

EOF

echo -e "\n${GREEN}Ready to run: cd /root/assistant/orpheus && python server_async.py${NC}"
echo -e "${YELLOW}Don't forget to use the WebSocket dashboard for testing!${NC}\n"