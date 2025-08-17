#!/usr/bin/env python3
"""
Orpheus TTS Service - Optimized FastAPI server with streaming support
Fixed max_tokens=64000 for 12+ minutes of audio generation
Based on server_async.py with all critical fixes applied
"""

import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, AsyncGenerator
import struct
import time
import torch
import gc
import os
import json
import yaml
import logging
from datetime import datetime
import numpy as np
from orpheus_tts import OrpheusModel
import argparse
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Request models
class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"
    temperature: float = 0.6
    top_p: float = 0.8
    repetition_penalty: float = 1.3
    max_tokens: Optional[int] = None  # Will use config default if not specified
    stream: bool = True
    return_format: str = "wav"

class OrpheusService:
    """Main TTS service with all optimizations and fixes"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self.load_config(config_path)
        self.model = None
        self.stats = {
            "requests_total": 0,
            "requests_success": 0,
            "requests_failed": 0,
            "total_audio_seconds": 0,
            "average_ttfb_ms": 0,
            "server_start_time": datetime.now().isoformat()
        }
        
    def load_config(self, config_path: str) -> dict:
        """Load configuration from YAML file"""
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration with FIXED max_tokens
            return {
                'model': {
                    'name': 'canopylabs/orpheus-tts-0.1-finetune-prod',
                    'max_model_len': 64000,
                    'max_tokens': 64000,  # FIXED: Was 2000, now 64000 for 12+ minutes!
                    'dtype': 'bfloat16'
                },
                'server': {
                    'host': '0.0.0.0',
                    'port': 8080,
                    'workers': 1
                },
                'audio': {
                    'sample_rate': 24000,
                    'channels': 1,
                    'bits_per_sample': 16
                },
                'generation': {
                    'temperature': 0.6,
                    'top_p': 0.8,
                    'repetition_penalty': 1.3
                }
            }
    
    async def initialize(self):
        """Initialize the model with all fixes applied"""
        logger.info("Initializing Orpheus TTS Service...")
        logger.info(f"Max tokens: {self.config['model']['max_tokens']} (~{self.config['model']['max_tokens']/83:.1f} seconds of audio)")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        try:
            # Initialize model - will use fixed config from HF cache
            self.model = OrpheusModel(
                model_name=self.config['model']['name'],
                dtype=getattr(torch, self.config['model']['dtype'], torch.bfloat16),
                max_model_len=self.config['model']['max_model_len']
            )
            
            logger.info("✅ Model loaded successfully")
            
            # Warmup
            await self.warmup()
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize model: {e}")
            logger.error("Run 'python orpheus_manager.py fix' to apply critical fixes")
            raise
    
    async def warmup(self):
        """Warmup the model with a short generation"""
        logger.info("Running warmup generation...")
        try:
            text = "Hello, this is a warmup test."
            async for _ in self.generate_stream(text, "tara", max_tokens=200):
                pass
            logger.info("✅ Warmup complete")
        except Exception as e:
            logger.warning(f"Warmup failed (non-critical): {e}")
    
    def create_wav_header(self, data_size: int = 0xFFFFFFFF) -> bytes:
        """Create WAV header for streaming"""
        sample_rate = self.config['audio']['sample_rate']
        channels = self.config['audio']['channels']
        bits_per_sample = self.config['audio']['bits_per_sample']
        
        byte_rate = sample_rate * channels * bits_per_sample // 8
        block_align = channels * bits_per_sample // 8
        
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
    
    async def generate_stream(
        self, 
        text: str, 
        voice: str = "tara",
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio stream with configurable max_tokens"""
        
        # Use defaults from config if not specified
        temperature = temperature or self.config['generation']['temperature']
        top_p = top_p or self.config['generation']['top_p']
        repetition_penalty = repetition_penalty or self.config['generation']['repetition_penalty']
        max_tokens = max_tokens or self.config['model']['max_tokens']  # Default to 64000!
        
        # Log generation parameters
        logger.info(f"Generating: voice={voice}, max_tokens={max_tokens} (~{max_tokens/83:.1f}s max audio)")
        
        start_time = time.time()
        first_chunk = True
        total_audio_seconds = 0
        
        try:
            # Generate audio
            generator = self.model.generate_speech(
                prompt=text,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,  # Using the FIXED value!
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258]
            )
            
            # Stream chunks
            for audio_chunk in generator:
                if first_chunk:
                    ttfb = (time.time() - start_time) * 1000
                    logger.info(f"TTFB: {ttfb:.1f}ms")
                    self.update_stats(ttfb_ms=ttfb)
                    first_chunk = False
                
                # Calculate audio duration (24kHz sample rate)
                chunk_samples = len(audio_chunk) // 2  # 16-bit audio
                chunk_duration = chunk_samples / self.config['audio']['sample_rate']
                total_audio_seconds += chunk_duration
                
                yield audio_chunk
            
            # Log final statistics
            total_time = time.time() - start_time
            logger.info(f"Generated {total_audio_seconds:.1f}s of audio in {total_time:.1f}s")
            logger.info(f"RTF: {total_time/total_audio_seconds:.2f}x")
            
            self.update_stats(audio_seconds=total_audio_seconds, success=True)
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            self.update_stats(success=False)
            raise
    
    def update_stats(self, ttfb_ms: float = None, audio_seconds: float = 0, success: bool = None):
        """Update server statistics"""
        if ttfb_ms is not None:
            # Update rolling average
            current_avg = self.stats["average_ttfb_ms"]
            count = self.stats["requests_total"]
            self.stats["average_ttfb_ms"] = (current_avg * count + ttfb_ms) / (count + 1) if count > 0 else ttfb_ms
        
        if audio_seconds > 0:
            self.stats["total_audio_seconds"] += audio_seconds
        
        if success is not None:
            self.stats["requests_total"] += 1
            if success:
                self.stats["requests_success"] += 1
            else:
                self.stats["requests_failed"] += 1

# Create FastAPI app
app = FastAPI(title="Orpheus TTS Service", version="2.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
service: Optional[OrpheusService] = None

@app.on_event("startup")
async def startup_event():
    """Initialize service on startup"""
    global service
    service = OrpheusService()
    await service.initialize()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if service and service.model:
        return JSONResponse({"status": "healthy", "model": service.config['model']['name']})
    return JSONResponse({"status": "initializing"}, status_code=503)

@app.get("/stats")
async def get_stats():
    """Get server statistics"""
    if service:
        return JSONResponse(service.stats)
    return JSONResponse({"error": "Service not initialized"}, status_code=503)

@app.post("/tts")
async def generate_tts(request: TTSRequest):
    """Generate TTS with streaming response"""
    if not service or not service.model:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Clean text
    text = request.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    # Log request with max_tokens info
    max_tokens = request.max_tokens or service.config['model']['max_tokens']
    logger.info(f"TTS request: {len(text)} chars, max_tokens={max_tokens}")
    
    async def audio_generator():
        """Generate audio with WAV header"""
        # Send WAV header first
        if request.return_format == "wav":
            yield service.create_wav_header()
        
        # Stream audio chunks
        async for chunk in service.generate_stream(
            text=text,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
            max_tokens=max_tokens
        ):
            yield chunk
    
    return StreamingResponse(
        audio_generator(),
        media_type="audio/wav" if request.return_format == "wav" else "audio/raw",
        headers={
            "X-Max-Tokens": str(max_tokens),
            "X-Max-Duration-Seconds": str(max_tokens / 83)
        }
    )

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time streaming"""
    await websocket.accept()
    
    try:
        while True:
            # Receive request
            data = await websocket.receive_json()
            
            # Extract parameters
            text = data.get("text", "")
            voice = data.get("voice", "tara")
            max_tokens = data.get("max_tokens", service.config['model']['max_tokens'])
            
            if not text:
                await websocket.send_json({"error": "Text cannot be empty"})
                continue
            
            # Send audio chunks
            chunk_count = 0
            async for audio_chunk in service.generate_stream(
                text=text,
                voice=voice,
                max_tokens=max_tokens
            ):
                # Convert to base64 for WebSocket transmission
                import base64
                audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                
                await websocket.send_json({
                    "type": "audio_chunk",
                    "chunk": chunk_count,
                    "data": audio_b64
                })
                chunk_count += 1
            
            # Send completion message
            await websocket.send_json({
                "type": "complete",
                "chunks": chunk_count
            })
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await websocket.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Orpheus TTS Service')
    parser.add_argument('--config', default='config.yaml', help='Path to config file')
    parser.add_argument('--host', default=None, help='Override host from config')
    parser.add_argument('--port', type=int, default=None, help='Override port from config')
    
    args = parser.parse_args()
    
    # Load config to get server settings
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        config = {'server': {'host': '0.0.0.0', 'port': 8080}}
    
    host = args.host or config['server']['host']
    port = args.port or config['server']['port']
    
    # Run server
    import uvicorn
    uvicorn.run(
        "orpheus_service:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )

if __name__ == "__main__":
    main()