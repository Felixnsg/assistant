#!/usr/bin/env python3
"""
Orpheus TTS Async Streaming Server - THE WORKING VERSION
This is the one that was working perfectly before cleanup
"""

import asyncio
import aiofiles
import aiohttp
from aiohttp import web
import aiohttp_cors
import time
import torch
import struct
import json
import os
import gc
import logging
from typing import AsyncGenerator, Optional
from concurrent.futures import ThreadPoolExecutor
from orpheus_tts import OrpheusModel
import numpy as np
from collections import deque
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrpheusAsyncServer:
    """High-performance async server that was WORKING"""
    
    def __init__(self):
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.request_semaphore = asyncio.Semaphore(10)
        self.model_lock = asyncio.Lock()
        
        # Configuration  
        self.config = {
            "model_name": os.getenv("MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod"),
            "max_model_len": int(os.getenv("MAX_MODEL_LEN", "64000")),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95")),
        }
        
    async def initialize(self):
        """Initialize model with async loading"""
        async with self.model_lock:
            if self.model is not None:
                return
                
            logger.info("Initializing Orpheus model...")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            try:
                # Load model - NO max_model_len parameter here!
                self.model = OrpheusModel(
                    model_name=self.config["model_name"],
                    dtype=torch.bfloat16
                )
                
                logger.info("✅ Model initialized successfully")
                
                # Warmup
                await self._warmup()
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize model: {e}")
                raise
    
    async def _warmup(self):
        """Async warmup"""
        logger.info("Running warmup...")
        warmup_text = "Hello, this is a test."
        
        try:
            async for _ in self.generate_stream_async(warmup_text, "tara", max_tokens=200):
                pass
        except Exception as e:
            logger.warning(f"Warmup failed: {e}")
        
        logger.info("✅ Warmup complete")
    
    async def generate_stream_async(
        self, 
        text: str, 
        voice: str = "tara",
        temperature: float = 0.6,
        top_p: float = 0.8,
        repetition_penalty: float = 1.3,
        max_tokens: int = 2000  # Back to original working limit
    ) -> AsyncGenerator[bytes, None]:
        """Async generator for audio streaming"""
        
        start_time = time.time()
        first_chunk_time = None
        
        try:
            # Use thread executor for blocking model call
            loop = asyncio.get_event_loop()
            
            # Create generator in thread
            def generate():
                return self.model.generate_speech(
                    prompt=text,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,  # Using 64000!
                    repetition_penalty=repetition_penalty,
                    stop_token_ids=[128258]  # RESTORED - this marks end of speech!
                )
            
            # Run generator in executor
            generator = await loop.run_in_executor(self.executor, generate)
            
            # Stream chunks
            def get_next_chunk():
                try:
                    return next(generator)
                except StopIteration:
                    return None
            
            while True:
                # Get next chunk in executor
                chunk = await loop.run_in_executor(self.executor, get_next_chunk)
                
                if chunk is None:
                    break
                
                if first_chunk_time is None:
                    first_chunk_time = time.time() - start_time
                    logger.info(f"TTFB: {first_chunk_time * 1000:.1f}ms")
                
                yield chunk
                
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

# Create global server instance
server = OrpheusAsyncServer()

# Create aiohttp app
app = web.Application()

# Configure CORS
cors = aiohttp_cors.setup(app, defaults={
    "*": aiohttp_cors.ResourceOptions(
        allow_credentials=True,
        expose_headers="*",
        allow_headers="*",
        allow_methods="*"
    )
})

async def startup(app):
    """Initialize on startup"""
    await server.initialize()

app.on_startup.append(startup)

# Routes
async def health(request):
    """Health check"""
    if server.model:
        return web.json_response({"status": "healthy"})
    return web.json_response({"status": "initializing"}, status=503)

async def generate_tts(request):
    """HTTP endpoint for TTS"""
    try:
        data = await request.json()
        text = data.get('text', '')
        voice = data.get('voice', 'tara')
        max_tokens = data.get('max_tokens', 64000)
        
        if not text:
            return web.json_response({"error": "Text required"}, status=400)
        
        # Generate audio
        audio_data = b""
        async for chunk in server.generate_stream_async(
            text=text,
            voice=voice,
            max_tokens=max_tokens
        ):
            audio_data += chunk
        
        # Return with WAV header
        wav_header = create_wav_header(len(audio_data))
        return web.Response(
            body=wav_header + audio_data,
            content_type='audio/wav'
        )
        
    except Exception as e:
        logger.error(f"TTS generation failed: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def websocket_handler(request):
    """WebSocket endpoint"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                text = data.get('text', '')
                voice = data.get('voice', 'tara')
                max_tokens = data.get('max_tokens', 64000)
                
                if not text:
                    await ws.send_json({"error": "Text required"})
                    continue
                
                # Log what we're generating
                logger.info(f"WebSocket TTS request: text_len={len(text)}, voice={voice}, max_tokens={max_tokens}")
                
                # Send audio chunks
                chunk_count = 0
                total_audio_bytes = 0
                async for audio_chunk in server.generate_stream_async(
                    text=text,
                    voice=voice,
                    max_tokens=max_tokens
                ):
                    total_audio_bytes += len(audio_chunk)
                    # Send as base64
                    import base64
                    audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                    
                    await ws.send_json({
                        "type": "audio_chunk",
                        "chunk": chunk_count,
                        "data": audio_b64
                    })
                    chunk_count += 1
                
                # Log completion stats
                audio_duration_sec = total_audio_bytes / 2 / 24000  # 16-bit audio at 24kHz
                logger.info(f"WebSocket complete: {chunk_count} chunks, {audio_duration_sec:.1f}s of audio, {total_audio_bytes/1024:.1f}KB")
                
                # Send completion
                await ws.send_json({
                    "type": "complete",
                    "chunks": chunk_count
                })
                
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                await ws.send_json({"error": str(e)})
                
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error(f'WebSocket error: {ws.exception()}')
            
    return ws

def create_wav_header(data_size):
    """Create WAV header"""
    sample_rate = 24000
    channels = 1
    bits_per_sample = 16
    
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

# Add routes
app.router.add_get('/health', health)
app.router.add_post('/tts', generate_tts)
app.router.add_get('/ws', websocket_handler)

# Add CORS to all routes
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    web.run_app(app, host='0.0.0.0', port=port)