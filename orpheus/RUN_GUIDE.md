# Orpheus TTS - Complete Running Guide

## Quick Start (After Cloning)

```bash
# 1. Initial setup (run once)
chmod +x setup.sh
sudo ./setup.sh

# 2. Start production server
python3 server.py

# Server is now running at http://0.0.0.0:8080
```

## Step-by-Step Guide

### 1. First Time Setup (Run Once)

```bash
# Clone the repository
git clone <your-repo-url>
cd orpheus

# Make scripts executable
chmod +x *.sh

# Run automated setup (installs everything)
sudo ./setup.sh
```

This will:
- Install all system dependencies
- Install Python packages (torch, vllm, orpheus-speech, etc.)
- Download and cache the models
- Create directories
- Configure environment
- Run initial tests

### 2. Manual Dependencies Installation (if setup.sh fails)

```bash
# Update system
apt-get update

# Install Python and pip
apt-get install -y python3-pip python3-dev

# Install PyTorch with CUDA
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Orpheus and core dependencies
pip3 install orpheus-speech
pip3 install vllm==0.7.3  # Important: use this specific version

# Install server dependencies
pip3 install -r requirements.txt

# Additional server packages
pip3 install fastapi uvicorn[standard] aiohttp aiohttp-cors psutil
```

### 3. Starting the Server (Choose One)

#### Option A: Production Server (Recommended)
```bash
# FastAPI with uvicorn - Best for production
python3 server.py

# Or run in background
nohup python3 server.py > server.log 2>&1 &

# Or use screen/tmux
tmux new -s orpheus
python3 server.py
# Ctrl+B then D to detach
```

#### Option B: Async Ultra-Low Latency Server
```bash
# For <200ms TTFB streaming
python3 server_async.py
```

#### Option C: Simple Flask Server (Development)
```bash
# Basic streaming server
python3 streaming_server.py
```

### 4. Verify Server is Running

```bash
# Check health
curl http://localhost:8080/health

# Test TTS generation
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, Orpheus is working!", "voice": "tara"}' \
  --output test.wav

# Play the audio (if you have a player)
play test.wav  # or ffplay test.wav
```

### 5. Monitor Performance

```bash
# In a new terminal, start the monitor
python3 monitor.py --url http://localhost:8080

# Or run a single test
python3 monitor.py --url http://localhost:8080 --test
```

### 6. Run Test Suite

```bash
# Basic inference test
python3 test_basic.py

# Benchmark performance
python3 benchmark.py

# Test emotions
python3 test_emotions.py

# Test all voices
python3 test_voice_clone.py
```

## API Usage Examples

### Basic TTS Generation
```bash
# Simple GET request
curl "http://localhost:8080/tts?text=Hello%20world&voice=tara" > output.wav

# POST with parameters
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This is a longer test with custom parameters.",
    "voice": "leo",
    "temperature": 0.7,
    "top_p": 0.9,
    "stream": true
  }' --output output.wav
```

### Using Different Voices
```bash
# Available voices: tara, leah, jess, leo, dan, mia, zac, zoe
for voice in tara leo mia; do
  curl -X POST http://localhost:8080/tts \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"Hello from $voice\", \"voice\": \"$voice\"}" \
    --output "${voice}.wav"
done
```

### Emotion Tags
```bash
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "<laugh> This is so funny! <sigh> But also tiring.",
    "voice": "tara"
  }' --output emotions.wav
```

### WebSocket Streaming (Advanced)
```python
# Python WebSocket client example
import asyncio
import websockets
import json

async def stream_tts():
    uri = "ws://localhost:8080/ws"
    async with websockets.connect(uri) as websocket:
        # Send request
        await websocket.send(json.dumps({
            "text": "Hello from WebSocket!",
            "voice": "tara"
        }))
        
        # Receive audio chunks
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            if data.get("type") == "complete":
                break
            # Process audio chunk here

asyncio.run(stream_tts())
```

## Troubleshooting

### CUDA Out of Memory
```bash
# Reduce max_model_len in server
export MAX_MODEL_LEN=1024
python3 server.py

# Or edit .env file
echo "MAX_MODEL_LEN=1024" >> .env
```

### Model Loading Issues
```bash
# Clear cache and re-download
rm -rf ~/.cache/huggingface
python3 -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('canopylabs/orpheus-tts-0.1-finetune-prod')"
```

### Port Already in Use
```bash
# Find and kill process using port 8080
lsof -i :8080
kill -9 <PID>

# Or use different port
PORT=8090 python3 server.py
```

### vLLM Version Issues
```bash
# Downgrade to stable version
pip3 uninstall vllm
pip3 install vllm==0.7.3
```

## Production Deployment

### Using tmux (Recommended)
```bash
# Create tmux session
tmux new -s orpheus

# Start server
python3 server.py

# Detach: Ctrl+B, then D
# Reattach: tmux attach -t orpheus
# List sessions: tmux ls
```

### Using systemd (Permanent Service)
```bash
# Create service (if root)
sudo cp orpheus-tts.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable orpheus-tts
sudo systemctl start orpheus-tts

# Check status
sudo systemctl status orpheus-tts

# View logs
sudo journalctl -u orpheus-tts -f
```

### Using Docker
```bash
# Build image
docker build -t orpheus-tts .

# Run container
docker run -d \
  --gpus all \
  -p 8080:8080 \
  -v $(pwd)/output:/app/output \
  --name orpheus \
  orpheus-tts

# Or use docker-compose
docker-compose up -d
```

## Performance Tuning

### For Best TTFB (<200ms)
```bash
# Use async server
python3 server_async.py

# Enable optimizations
export GPU_MEMORY_UTILIZATION=0.95
export ENABLE_PREFIX_CACHING=true
export ENABLE_CHUNKED_PREFILL=true
```

### For Maximum Throughput
```bash
# Increase batch size
export MAX_NUM_SEQS=256
export MAX_MODEL_LEN=2048
```

### GPU Optimization
```bash
# Enable persistence mode
sudo nvidia-smi -pm 1

# Set max performance
sudo nvidia-smi -ac 5001,1980  # For RTX 4090
```

## Monitoring & Logs

### Real-time Monitoring
```bash
# Terminal 1: Server
python3 server.py

# Terminal 2: Monitor
python3 monitor.py --url http://localhost:8080

# Terminal 3: GPU monitoring
watch -n 1 nvidia-smi

# Terminal 4: System resources
htop
```

### Check Logs
```bash
# Server logs
tail -f server.log

# Monitor history
cat monitor_history.json | jq .

# Benchmark results
cat output/benchmark_results.json | jq .
```

## Test Remote Access

If your vast.ai instance has public IP:

```bash
# From your local machine
curl http://<VAST_AI_IP>:8080/health

# Stream audio
curl "http://<VAST_AI_IP>:8080/tts?text=Remote%20test&voice=tara" > remote.wav
```

## Stopping the Server

```bash
# If running in foreground
Ctrl+C

# If running in background
ps aux | grep server.py
kill <PID>

# If using tmux
tmux attach -t orpheus
Ctrl+C

# If using systemd
sudo systemctl stop orpheus-tts
```

## Quick Commands Reference

```bash
# Start server
python3 server.py

# Test health
curl http://localhost:8080/health

# Generate TTS
curl -X POST http://localhost:8080/tts -H "Content-Type: application/json" -d '{"text": "Test", "voice": "tara"}' -o test.wav

# Monitor
python3 monitor.py --url http://localhost:8080

# Benchmark
python3 benchmark.py

# View GPU
nvidia-smi

# Check processes
ps aux | grep python

# View logs
tail -f *.log
```

## Success Indicators

✅ Server starts without errors
✅ Health check returns "healthy"
✅ TTFB < 500ms (check monitor output)
✅ GPU memory usage ~12-20GB (nvidia-smi)
✅ Can generate audio files
✅ No CUDA OOM errors

## Support

If you encounter issues:
1. Check GPU is available: `nvidia-smi`
2. Verify CUDA: `python3 -c "import torch; print(torch.cuda.is_available())"`
3. Check logs: `tail -f server.log`
4. Run monitor test: `python3 monitor.py --test`
5. Try reducing model size: `MAX_MODEL_LEN=1024 python3 server.py`