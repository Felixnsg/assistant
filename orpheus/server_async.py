#!/usr/bin/env python3
"""
Orpheus TTS Async Streaming Server - Optimized for Ultra-Low Latency
Uses asyncio and vLLM's native async capabilities for maximum performance
"""

import asyncio
import aiofiles
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

# Performance monitoring
class PerformanceMonitor:
    def __init__(self, window_size=100):
        self.ttfb_history = deque(maxlen=window_size)
        self.throughput_history = deque(maxlen=window_size)
        self.request_count = 0
        self.start_time = time.time()
        
    def record_ttfb(self, ttfb_ms):
        self.ttfb_history.append(ttfb_ms)
    
    def record_throughput(self, audio_seconds, generation_time):
        if generation_time > 0:
            rtf = generation_time / audio_seconds
            self.throughput_history.append(rtf)
    
    def get_stats(self):
        uptime = time.time() - self.start_time
        return {
            "uptime_seconds": uptime,
            "requests_total": self.request_count,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
            "ttfb_ms": {
                "current": list(self.ttfb_history)[-1] if self.ttfb_history else 0,
                "min": min(self.ttfb_history) if self.ttfb_history else 0,
                "max": max(self.ttfb_history) if self.ttfb_history else 0,
                "avg": sum(self.ttfb_history) / len(self.ttfb_history) if self.ttfb_history else 0,
                "p50": np.percentile(list(self.ttfb_history), 50) if self.ttfb_history else 0,
                "p95": np.percentile(list(self.ttfb_history), 95) if self.ttfb_history else 0,
                "p99": np.percentile(list(self.ttfb_history), 99) if self.ttfb_history else 0,
            },
            "rtf": {
                "avg": sum(self.throughput_history) / len(self.throughput_history) if self.throughput_history else 0,
                "min": min(self.throughput_history) if self.throughput_history else 0,
                "max": max(self.throughput_history) if self.throughput_history else 0,
            },
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "gpu_memory_mb": torch.cuda.memory_allocated() / 1024**2 if torch.cuda.is_available() else 0,
            }
        }

class OrpheusAsyncServer:
    """High-performance async server with connection pooling and optimizations"""
    
    def __init__(self):
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.monitor = PerformanceMonitor()
        self.request_semaphore = asyncio.Semaphore(10)  # Limit concurrent requests
        self.model_lock = asyncio.Lock()
        
        # Configuration
        self.config = {
            "model_name": os.getenv("MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod"),
            "max_model_len": int(os.getenv("MAX_MODEL_LEN", "2048")),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95")),
            "enable_prefix_caching": os.getenv("ENABLE_PREFIX_CACHING", "true").lower() == "true",
            "enable_chunked_prefill": os.getenv("ENABLE_CHUNKED_PREFILL", "true").lower() == "true",
            "max_num_seqs": int(os.getenv("MAX_NUM_SEQS", "256")),
            "tensor_parallel_size": int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
        }
        
    async def initialize(self):
        """Initialize model with async loading"""
        async with self.model_lock:
            if self.model is not None:
                return
                
            logger.info("Initializing Orpheus model with async optimizations...")
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
            
            try:
                # Load model with vLLM optimizations
                # vLLM parameters go into engine_kwargs
                self.model = OrpheusModel(
                    model_name=self.config["model_name"],
                    dtype=torch.bfloat16,
                    max_model_len=self.config["max_model_len"],
                    trust_remote_code=True,
                    gpu_memory_utilization=self.config["gpu_memory_utilization"],
                    enable_prefix_caching=self.config["enable_prefix_caching"],
                    enable_chunked_prefill=self.config["enable_chunked_prefill"],
                    max_num_seqs=self.config["max_num_seqs"],
                    tensor_parallel_size=self.config["tensor_parallel_size"],
                    disable_log_stats=True,  # Reduce overhead
                    enforce_eager=False,  # Allow CUDA graphs
                )
                
                logger.info("✅ Model initialized successfully")
                
                # Warmup
                await self._warmup()
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize model: {e}")
                raise
    
    async def _warmup(self):
        """Async warmup with multiple samples"""
        logger.info("Running warmup...")
        warmup_texts = [
            "Hello.",
            "This is a test of the system.",
            "The quick brown fox jumps over the lazy dog.",
        ]
        
        for text in warmup_texts:
            try:
                async for _ in self.generate_stream_async(text, "tara"):
                    pass
            except Exception as e:
                logger.warning(f"Warmup sample failed: {e}")
        
        logger.info("✅ Warmup complete")
    
    async def generate_stream_async(
        self, 
        text: str, 
        voice: str = "tara",
        temperature: float = 0.6,
        top_p: float = 0.8,
        repetition_penalty: float = 1.3,
        max_tokens: int = 2000
    ) -> AsyncGenerator[bytes, None]:
        """Async generator for audio streaming with monitoring"""
        
        start_time = time.time()
        first_chunk_time = None
        total_bytes = 0
        chunk_count = 0
        
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
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty,
                    stop_token_ids=[128258]
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
                    ttfb_ms = first_chunk_time * 1000
                    self.monitor.record_ttfb(ttfb_ms)
                    logger.info(f"TTFB: {ttfb_ms:.1f}ms")
                
                chunk_count += 1
                total_bytes += len(chunk)
                
                yield chunk
            
            # Record statistics
            total_time = time.time() - start_time
            audio_duration = total_bytes / (2 * 24000)  # 16-bit, 24kHz
            self.monitor.record_throughput(audio_duration, total_time)
            self.monitor.request_count += 1
            
            logger.info(f"Generated {audio_duration:.2f}s in {total_time:.2f}s "
                       f"(RTF: {total_time/audio_duration:.2f}x, {chunk_count} chunks)")
            
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def create_wav_header(self, sample_rate=24000):
        """Create streaming WAV header"""
        return struct.pack(
            '<4sI4s4sIHHIIHH4sI',
            b'RIFF', 0xFFFFFFFF, b'WAVE',
            b'fmt ', 16, 1, 1,
            sample_rate, sample_rate * 2, 2, 16,
            b'data', 0xFFFFFFFF
        )

# Global server instance
server = OrpheusAsyncServer()

# Route handlers
async def handle_index(request):
    """API documentation"""
    return web.json_response({
        "name": "Orpheus TTS Async Streaming Server",
        "version": "2.0.0",
        "endpoints": {
            "/tts": "Stream TTS audio (GET/POST)",
            "/health": "Health check",
            "/stats": "Performance statistics",
            "/voices": "List available voices",
        },
        "features": [
            "Ultra-low latency streaming (<200ms TTFB)",
            "Async request handling",
            "Connection pooling",
            "Performance monitoring",
            "GPU optimization"
        ]
    })

async def handle_health(request):
    """Health check with detailed status"""
    health = {
        "status": "healthy",
        "model_loaded": server.model is not None,
        "timestamp": time.time(),
    }
    
    if torch.cuda.is_available():
        health.update({
            "gpu": {
                "name": torch.cuda.get_device_name(0),
                "memory_allocated_mb": torch.cuda.memory_allocated() / 1024**2,
                "memory_cached_mb": torch.cuda.memory_reserved() / 1024**2,
                "utilization": torch.cuda.utilization(),
            }
        })
    
    return web.json_response(health)

async def handle_stats(request):
    """Performance statistics"""
    return web.json_response(server.monitor.get_stats())

async def handle_voices(request):
    """List available voices"""
    return web.json_response({
        "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
        "default": "tara"
    })

async def handle_tts(request):
    """Main TTS streaming endpoint"""
    
    # Get parameters
    if request.method == "POST":
        data = await request.json()
        text = data.get("text", "")
        voice = data.get("voice", "tara")
        temperature = data.get("temperature", 0.6)
        top_p = data.get("top_p", 0.8)
        repetition_penalty = data.get("repetition_penalty", 1.3)
    else:
        text = request.query.get("text", "")
        voice = request.query.get("voice", "tara")
        temperature = float(request.query.get("temperature", "0.6"))
        top_p = float(request.query.get("top_p", "0.8"))
        repetition_penalty = float(request.query.get("repetition_penalty", "1.3"))
    
    if not text:
        return web.json_response({"error": "Text is required"}, status=400)
    
    # Limit concurrent requests
    async with server.request_semaphore:
        # Create streaming response
        response = web.StreamResponse(
            status=200,
            headers={
                'Content-Type': 'audio/wav',
                'Cache-Control': 'no-cache',
                'X-Accel-Buffering': 'no',  # Disable nginx buffering
            }
        )
        
        await response.prepare(request)
        
        try:
            # Send WAV header
            await response.write(server.create_wav_header())
            
            # Stream audio chunks
            async for chunk in server.generate_stream_async(
                text, voice, temperature, top_p, repetition_penalty
            ):
                await response.write(chunk)
                
                # Flush to ensure low latency
                # await response.drain()
            
            await response.write_eof()
            
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            # Connection likely closed
        
        return response

async def handle_websocket(request):
    """WebSocket endpoint for bidirectional streaming"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    logger.info("WebSocket connection opened")
    
    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                data = json.loads(msg.data)
                
                # Start generation
                text = data.get("text", "")
                voice = data.get("voice", "tara")
                
                if not text:
                    await ws.send_json({"error": "Text is required"})
                    continue
                
                # Stream audio chunks
                chunk_id = 0
                async for audio_chunk in server.generate_stream_async(text, voice):
                    await ws.send_bytes(audio_chunk)
                    chunk_id += 1
                
                # Send completion marker
                await ws.send_json({"type": "complete", "chunks": chunk_id})
                
            elif msg.type == aiohttp.WSMsgType.ERROR:
                logger.error(f'WebSocket error: {ws.exception()}')
                
    except Exception as e:
        logger.error(f"WebSocket handler error: {e}")
    finally:
        logger.info("WebSocket connection closed")
        
    return ws

async def init_app():
    """Initialize the application"""
    app = web.Application()
    
    # Setup CORS
    cors = aiohttp_cors.setup(app, defaults={
        "*": aiohttp_cors.ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add routes
    app.router.add_get('/', handle_index)
    app.router.add_get('/health', handle_health)
    app.router.add_get('/stats', handle_stats)
    app.router.add_get('/voices', handle_voices)
    app.router.add_get('/tts', handle_tts)
    app.router.add_post('/tts', handle_tts)
    app.router.add_get('/ws', handle_websocket)
    
    # Configure CORS for all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    # Initialize model on startup
    async def on_startup(app):
        await server.initialize()
    
    # Cleanup on shutdown
    async def on_cleanup(app):
        logger.info("Cleaning up...")
        server.executor.shutdown(wait=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    
    return app

if __name__ == '__main__':
    print("=" * 60)
    print("ORPHEUS TTS ASYNC STREAMING SERVER")
    print("=" * 60)
    print("Optimized for ultra-low latency streaming")
    print("Target TTFB: <200ms on RTX 4090")
    print("=" * 60)
    
    # Run server
    web.run_app(
        init_app(),
        host='0.0.0.0',
        port=8080,
        access_log=None,  # Disable for performance
        print=lambda x: logger.info(x)
    )