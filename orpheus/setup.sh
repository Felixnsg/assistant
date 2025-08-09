#!/bin/bash
# Orpheus TTS Server Setup Script for vast.ai RTX 4090
# Automated deployment and configuration

set -e  # Exit on error

echo "============================================"
echo "ORPHEUS TTS SERVER SETUP"
echo "============================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running on GPU instance
check_gpu() {
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi
        print_status "GPU detected"
    else
        print_warning "No GPU detected - performance will be limited"
    fi
}

# Install system dependencies
install_system_deps() {
    echo "Installing system dependencies..."
    apt-get update
    apt-get install -y \
        python3-pip \
        python3-dev \
        git \
        wget \
        curl \
        htop \
        nvtop \
        ffmpeg \
        libsndfile1 \
        sox \
        build-essential \
        tmux \
        screen
    print_status "System dependencies installed"
}

# Install Python dependencies
install_python_deps() {
    echo "Installing Python dependencies..."
    
    # Upgrade pip
    pip3 install --upgrade pip
    
    # Install PyTorch with CUDA support
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    # Install Orpheus and dependencies
    pip3 install orpheus-speech
    
    # Downgrade vLLM if needed
    pip3 install vllm==0.7.3
    
    # Install server dependencies
    pip3 install \
        fastapi \
        uvicorn[standard] \
        aiohttp \
        aiofiles \
        aiohttp-cors \
        pydantic \
        python-multipart \
        python-dotenv \
        pyyaml \
        psutil \
        soundfile \
        scipy \
        numpy \
        transformers \
        snac
    
    print_status "Python dependencies installed"
}

# Download and cache model
cache_model() {
    echo "Downloading and caching Orpheus model..."
    python3 << EOF
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
print(f"Downloading {model_name}...")

try:
    # Download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Tokenizer downloaded")
    
    # Download model (this will cache it)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    print("Model downloaded and cached")
    
    # Also download SNAC model
    from snac import SNAC
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz")
    print("SNAC decoder downloaded")
    
    print("✓ All models cached successfully")
except Exception as e:
    print(f"Error caching models: {e}")
    print("Models will be downloaded on first run")
EOF
    
    print_status "Model caching complete"
}

# Setup directories
setup_directories() {
    echo "Setting up directories..."
    mkdir -p output
    mkdir -p logs
    mkdir -p /tmp/model_cache
    mkdir -p /tmp/audio_cache
    print_status "Directories created"
}

# Configure environment
setup_environment() {
    echo "Configuring environment..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        cp .env.example .env 2>/dev/null || cat > .env << EOF
HOST=0.0.0.0
PORT=8080
MODEL_NAME=canopylabs/orpheus-tts-0.1-finetune-prod
MAX_MODEL_LEN=2048
GPU_MEMORY_UTILIZATION=0.95
ENABLE_PREFIX_CACHING=true
WARMUP_ON_START=true
CUDA_VISIBLE_DEVICES=0
PYTHONUNBUFFERED=1
EOF
        print_status "Environment file created"
    else
        print_warning ".env file already exists"
    fi
    
    # Set CUDA environment variables
    export CUDA_VISIBLE_DEVICES=0
    export CUDA_LAUNCH_BLOCKING=0
    export PYTHONUNBUFFERED=1
    
    print_status "Environment configured"
}

# Create systemd service
create_service() {
    echo "Creating systemd service..."
    
    cat > /etc/systemd/system/orpheus-tts.service << EOF
[Unit]
Description=Orpheus TTS Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$(pwd)
Environment="PATH=/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
Environment="CUDA_VISIBLE_DEVICES=0"
Environment="PYTHONUNBUFFERED=1"
ExecStart=/usr/bin/python3 $(pwd)/server.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    systemctl daemon-reload
    print_status "Systemd service created"
}

# Test server
test_server() {
    echo "Testing server..."
    
    # Start server in background
    python3 server.py &
    SERVER_PID=$!
    
    # Wait for server to start
    sleep 10
    
    # Test health endpoint
    if curl -s http://localhost:8080/health | grep -q "healthy"; then
        print_status "Server health check passed"
    else
        print_error "Server health check failed"
    fi
    
    # Test TTS generation
    curl -X POST http://localhost:8080/tts \
        -H "Content-Type: application/json" \
        -d '{"text": "Hello, Orpheus TTS is working!", "voice": "tara"}' \
        --output test_output.wav
    
    if [ -f test_output.wav ]; then
        print_status "TTS generation test passed"
        rm test_output.wav
    else
        print_error "TTS generation test failed"
    fi
    
    # Kill test server
    kill $SERVER_PID 2>/dev/null || true
}

# Setup monitoring
setup_monitoring() {
    echo "Setting up monitoring..."
    
    # Create monitoring script
    cat > start_monitor.sh << 'EOF'
#!/bin/bash
python3 monitor.py --url http://localhost:8080 --interval 30
EOF
    chmod +x start_monitor.sh
    
    print_status "Monitoring setup complete"
}

# Performance tuning
tune_performance() {
    echo "Applying performance tuning..."
    
    # GPU persistence mode
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi -pm 1
        print_status "GPU persistence mode enabled"
    fi
    
    # Increase file descriptor limits
    ulimit -n 65536
    
    # TCP tuning for low latency
    if [ -w /proc/sys/net/core/rmem_max ]; then
        echo 134217728 > /proc/sys/net/core/rmem_max
        echo 134217728 > /proc/sys/net/core/wmem_max
        echo 65536 > /proc/sys/net/core/netdev_max_backlog
        print_status "Network tuning applied"
    fi
    
    print_status "Performance tuning complete"
}

# Main setup flow
main() {
    echo "Starting Orpheus TTS setup..."
    echo "Target: vast.ai RTX 4090 instance"
    echo ""
    
    # Check if running as root for system changes
    if [ "$EUID" -ne 0 ]; then 
        print_warning "Not running as root - some optimizations may be skipped"
    fi
    
    # Run setup steps
    check_gpu
    install_system_deps
    install_python_deps
    setup_directories
    setup_environment
    cache_model
    
    # Optional: Create service if root
    if [ "$EUID" -eq 0 ]; then
        create_service
    fi
    
    setup_monitoring
    tune_performance
    test_server
    
    echo ""
    echo "============================================"
    echo "SETUP COMPLETE!"
    echo "============================================"
    echo ""
    echo "To start the server:"
    echo "  Production (FastAPI): python3 server.py"
    echo "  Async (aiohttp): python3 server_async.py"
    echo "  Development (Flask): python3 streaming_server.py"
    echo ""
    echo "To monitor:"
    echo "  python3 monitor.py --url http://localhost:8080"
    echo ""
    echo "To test:"
    echo "  python3 test_basic.py"
    echo "  python3 benchmark.py"
    echo ""
    echo "Server will be available at:"
    echo "  http://0.0.0.0:8080"
    echo ""
    print_status "Setup completed successfully!"
}

# Run main function
main "$@"