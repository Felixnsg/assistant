# Orpheus TTS Server Docker Image for vast.ai
# Optimized for RTX 4090 with CUDA 12.1

FROM pytorch/pytorch:2.0.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    MODEL_CACHE_DIR=/models \
    HF_HOME=/models/huggingface

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    htop \
    tmux \
    ffmpeg \
    libsndfile1 \
    sox \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    orpheus-speech \
    vllm==0.7.3 \
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
    uvloop

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/output /app/logs /models /tmp/audio_cache

# Pre-download models (optional - speeds up first run)
# Uncomment if you want to bake models into the image
# RUN python3 -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoTokenizer.from_pretrained('canopylabs/orpheus-tts-0.1-finetune-prod'); \
#     AutoModelForCausalLM.from_pretrained('canopylabs/orpheus-tts-0.1-finetune-prod', torch_dtype='auto'); \
#     from snac import SNAC; SNAC.from_pretrained('hubertsiuzdak/snac_24khz')"

# Make scripts executable
RUN chmod +x setup.sh

# Expose ports
EXPOSE 8080 22

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command - can be overridden
CMD ["python3", "server.py"]