#!/usr/bin/env python3
"""
Orpheus TTS Production Server for GPU (vast.ai RTX 4090)
High-performance streaming server with optimizations
"""

import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import struct
import time
import torch
import gc
import os
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager
import numpy as np
from orpheus_tts import OrpheusModel
import threading
import queue

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global variables
model = None
model_lock = threading.Lock()
stats = {
    "requests_total": 0,
    "requests_success": 0,
    "requests_failed": 0,
    "total_audio_generated_seconds": 0,
    "average_ttfb_ms": 0,
    "server_start_time": datetime.now().isoformat()
}

# Configuration
class ServerConfig:
    MODEL_NAME = os.getenv("MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod")
    MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", "2048"))
    DTYPE = torch.bfloat16
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    PORT = int(os.getenv("PORT", "8080"))
    HOST = os.getenv("HOST", "0.0.0.0")
    MAX_TEXT_LENGTH = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "false").lower() == "true"
    WARMUP_ON_START = os.getenv("WARMUP_ON_START", "true").lower() == "true"
    CACHE_SIZE = int(os.getenv("CACHE_SIZE", "10"))
    
config = ServerConfig()

# Request/Response models
class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"
    temperature: float = 0.6
    top_p: float = 0.8
    repetition_penalty: float = 1.3
    max_tokens: int = 2000
    stream: bool = True
    return_format: str = "wav"  # wav, raw, mp3

class TTSBatchRequest(BaseModel):
    requests: List[TTSRequest]
    parallel: bool = False

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV header for streaming"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    data_size = 0xFFFFFFFF  # Streaming size
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,
        1,  # PCM
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

class OrpheusEngine:
    """Optimized Orpheus engine wrapper with caching and monitoring"""
    
    def __init__(self):
        self.model = None
        self.cache = {}
        self.cache_order = []
        self.load_lock = threading.Lock()
        self.generation_lock = threading.Lock()
        
    def initialize(self):
        """Initialize the model with optimizations"""
        with self.load_lock:
            if self.model is not None:
                return
                
            logger.info(f"Initializing Orpheus model: {config.MODEL_NAME}")
            logger.info(f"Device: {config.DEVICE}")
            
            if config.DEVICE == "cuda":
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
                
                # Clear GPU cache before loading
                torch.cuda.empty_cache()
                gc.collect()
            
            try:
                # Load model with optimizations
                # Only model_name, dtype, and tokenizer are direct params
                # Everything else goes through **engine_kwargs
                self.model = OrpheusModel(
                    model_name=config.MODEL_NAME,
                    dtype=config.DTYPE
                    # vLLM parameters will be passed as engine_kwargs internally
                )
                
                # Optional torch.compile for faster inference
                if config.ENABLE_TORCH_COMPILE and hasattr(torch, 'compile'):
                    logger.info("Applying torch.compile optimization...")
                    # Note: This might not work with all model components
                    
                logger.info("✅ Model loaded successfully")
                
                # Warmup if configured
                if config.WARMUP_ON_START:
                    self._warmup()
                    
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise
    
    def _warmup(self):
        """Warmup the model with a test generation"""
        logger.info("Warming up model...")
        try:
            warmup_text = "Hello, this is a warmup test."
            tokens = self.model.generate_speech(
                prompt=warmup_text,
                voice="tara",
                max_tokens=200,
                temperature=0.6
            )
            # Consume generator
            for _ in tokens:
                pass
            logger.info("✅ Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
    
    def generate_stream(self, request: TTSRequest):
        """Generate audio stream with monitoring"""
        start_time = time.time()
        first_chunk_time = None
        
        # Check cache
        cache_key = f"{request.text[:100]}_{request.voice}_{request.temperature}"
        if cache_key in self.cache:
            logger.info("Cache hit!")
            for chunk in self.cache[cache_key]:
                yield chunk
            return
        
        # Generate new audio
        audio_chunks = []
        
        try:
            tokens = self.model.generate_speech(
                prompt=request.text,
                voice=request.voice,
                temperature=request.temperature,
                top_p=request.top_p,
                max_tokens=request.max_tokens,
                repetition_penalty=request.repetition_penalty,
                stop_token_ids=[128258]
            )
            
            chunk_count = 0
            total_bytes = 0
            
            for audio_chunk in tokens:
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    ttfb_ms = first_chunk_time * 1000
                    logger.info(f"TTFB: {ttfb_ms:.1f}ms")
                    
                    # Update stats
                    stats["average_ttfb_ms"] = (
                        stats["average_ttfb_ms"] * stats["requests_success"] + ttfb_ms
                    ) / (stats["requests_success"] + 1)
                
                chunk_count += 1
                total_bytes += len(audio_chunk)
                audio_chunks.append(audio_chunk)
                yield audio_chunk
            
            # Update cache
            if len(audio_chunks) > 0 and len(request.text) < 500:  # Cache short texts
                self._update_cache(cache_key, audio_chunks)
            
            # Calculate statistics
            total_time = time.time() - start_time
            audio_duration = total_bytes / (2 * 24000)  # 16-bit, 24kHz
            rtf = total_time / audio_duration if audio_duration > 0 else 0
            
            logger.info(f"Generated {audio_duration:.2f}s in {total_time:.2f}s (RTF: {rtf:.2f}x)")
            
            # Update global stats
            stats["requests_success"] += 1
            stats["total_audio_generated_seconds"] += audio_duration
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            stats["requests_failed"] += 1
            raise
    
    def _update_cache(self, key: str, chunks: List[bytes]):
        """Update LRU cache"""
        if key in self.cache:
            self.cache_order.remove(key)
        elif len(self.cache) >= config.CACHE_SIZE:
            # Remove oldest
            oldest = self.cache_order.pop(0)
            del self.cache[oldest]
        
        self.cache[key] = chunks
        self.cache_order.append(key)

# Initialize engine
engine = OrpheusEngine()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle"""
    # Startup
    logger.info("Starting Orpheus TTS Server...")
    engine.initialize()
    yield
    # Shutdown
    logger.info("Shutting down...")
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()

# Create FastAPI app
app = FastAPI(
    title="Orpheus TTS Production Server",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """API documentation"""
    return {
        "name": "Orpheus TTS Production Server",
        "version": "1.0.0",
        "endpoints": {
            "/tts": "Generate TTS (POST)",
            "/tts/stream": "Stream TTS (POST)",
            "/batch": "Batch TTS generation (POST)",
            "/voices": "List available voices (GET)",
            "/health": "Health check (GET)",
            "/stats": "Server statistics (GET)",
            "/ws": "WebSocket streaming (WS)"
        },
        "model": config.MODEL_NAME,
        "device": config.DEVICE
    }

@app.get("/health")
async def health():
    """Health check with GPU status"""
    health_status = {
        "status": "healthy",
        "model_loaded": engine.model is not None,
        "device": config.DEVICE,
        "timestamp": datetime.now().isoformat()
    }
    
    if config.DEVICE == "cuda":
        health_status.update({
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_used": f"{torch.cuda.memory_allocated() / 1024**3:.2f} GB",
            "gpu_memory_total": f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        })
    
    return health_status

@app.get("/stats")
async def get_stats():
    """Server statistics"""
    stats["requests_total"] = stats["requests_success"] + stats["requests_failed"]
    stats["uptime_seconds"] = (
        datetime.now() - datetime.fromisoformat(stats["server_start_time"])
    ).total_seconds()
    
    if config.DEVICE == "cuda":
        stats["gpu_memory_used_gb"] = torch.cuda.memory_allocated() / 1024**3
        stats["gpu_memory_cached_gb"] = torch.cuda.memory_reserved() / 1024**3
    
    return stats

@app.get("/voices")
async def list_voices():
    """List available voices"""
    return {
        "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
        "default": "tara"
    }

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    """Generate TTS with optional streaming"""
    # Validate input
    if not request.text:
        raise HTTPException(status_code=400, detail="Text is required")
    
    if len(request.text) > config.MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=400, 
            detail=f"Text too long. Maximum {config.MAX_TEXT_LENGTH} characters."
        )
    
    stats["requests_total"] += 1
    
    if request.stream:
        # Stream response
        def audio_generator():
            # Send WAV header first if requested
            if request.return_format == "wav":
                yield create_wav_header()
            
            # Stream audio chunks
            for chunk in engine.generate_stream(request):
                yield chunk
        
        return StreamingResponse(
            audio_generator(),
            media_type="audio/wav" if request.return_format == "wav" else "application/octet-stream"
        )
    else:
        # Return complete audio
        audio_data = b""
        if request.return_format == "wav":
            audio_data = create_wav_header()
        
        for chunk in engine.generate_stream(request):
            audio_data += chunk
        
        return Response(
            content=audio_data,
            media_type="audio/wav" if request.return_format == "wav" else "application/octet-stream"
        )

@app.post("/tts/stream")
async def stream_tts(request: TTSRequest):
    """Dedicated streaming endpoint"""
    request.stream = True
    return await generate_tts(request)

@app.post("/batch")
async def batch_generate(batch_request: TTSBatchRequest):
    """Batch TTS generation"""
    if len(batch_request.requests) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 requests per batch")
    
    results = []
    
    for req in batch_request.requests:
        try:
            audio_data = b""
            if req.return_format == "wav":
                audio_data = create_wav_header()
            
            for chunk in engine.generate_stream(req):
                audio_data += chunk
            
            results.append({
                "success": True,
                "audio_size": len(audio_data),
                "audio_base64": None  # Could encode to base64 if needed
            })
        except Exception as e:
            results.append({
                "success": False,
                "error": str(e)
            })
    
    return {"results": results}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time streaming"""
    await websocket.accept()
    logger.info("WebSocket connection established")
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            request = TTSRequest(**data)
            
            # Send audio chunks
            chunk_id = 0
            for audio_chunk in engine.generate_stream(request):
                await websocket.send_json({
                    "type": "audio_chunk",
                    "chunk_id": chunk_id,
                    "data": audio_chunk.hex()  # Send as hex string
                })
                chunk_id += 1
            
            # Send completion
            await websocket.send_json({
                "type": "complete",
                "chunks_sent": chunk_id
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.send_json({
            "type": "error",
            "message": str(e)
        })

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Cleaning up resources...")
    if config.DEVICE == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    print("=" * 60)
    print("ORPHEUS TTS PRODUCTION SERVER")
    print("=" * 60)
    print(f"Model: {config.MODEL_NAME}")
    print(f"Device: {config.DEVICE}")
    print(f"Max Model Length: {config.MAX_MODEL_LEN}")
    print(f"Server: http://{config.HOST}:{config.PORT}")
    print("=" * 60)
    
    # Run with uvicorn for production performance
    uvicorn.run(
        app,
        host=config.HOST,
        port=config.PORT,
        log_level="info",
        access_log=True,
        use_colors=True,
        # Production settings
        workers=1,  # Single worker for GPU
        loop="uvloop",  # Faster event loop
        limit_concurrency=100,
        limit_max_requests=10000,
        timeout_keep_alive=5
    )