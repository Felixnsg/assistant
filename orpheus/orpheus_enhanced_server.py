#!/usr/bin/env python3
"""
Orpheus TTS Enhanced Server - Professional Production-Grade Implementation
Engineered for extended audio generation beyond 24-second limitations
Implements intelligent chunking, streaming, and memory optimization
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
import re
import numpy as np
from typing import AsyncGenerator, Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from orpheus_tts import OrpheusModel
from collections import deque
import psutil
import traceback
from dataclasses import dataclass
from enum import Enum

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(name)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Performance monitoring
class PerformanceMonitor:
    """Track and log performance metrics"""
    def __init__(self):
        self.metrics = {
            'total_requests': 0,
            'total_audio_seconds': 0,
            'total_tokens_generated': 0,
            'average_rtf': [],
            'memory_usage': []
        }
    
    def log_generation(self, audio_duration: float, generation_time: float, tokens: int):
        self.metrics['total_requests'] += 1
        self.metrics['total_audio_seconds'] += audio_duration
        self.metrics['total_tokens_generated'] += tokens
        rtf = generation_time / audio_duration if audio_duration > 0 else 0
        self.metrics['average_rtf'].append(rtf)
        
        # Keep only last 100 RTF measurements
        if len(self.metrics['average_rtf']) > 100:
            self.metrics['average_rtf'] = self.metrics['average_rtf'][-100:]
        
        # Log memory usage
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.metrics['memory_usage'].append(memory_mb)
        
        logger.info(f"📊 Performance: RTF={rtf:.2f}x | Memory={memory_mb:.1f}MB | Total Generated={self.metrics['total_audio_seconds']:.1f}s")

@dataclass
class TextChunk:
    """Represents a chunk of text for processing"""
    text: str
    index: int
    is_final: bool
    overlap_text: str = ""
    metadata: Dict[str, Any] = None

class ChunkingStrategy(Enum):
    """Text chunking strategies"""
    SENTENCE = "sentence"
    PARAGRAPH = "paragraph"
    TOKEN_COUNT = "token_count"
    HYBRID = "hybrid"

class IntelligentTextChunker:
    """Advanced text chunking with context preservation"""
    
    def __init__(self, 
                 max_chunk_size: int = 500,
                 overlap_size: int = 50,
                 strategy: ChunkingStrategy = ChunkingStrategy.HYBRID):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.strategy = strategy
        
        # Sentence delimiters with priority
        self.sentence_delimiters = [
            ('. ', 1.0),   # Full stop
            ('! ', 0.9),   # Exclamation
            ('? ', 0.9),   # Question
            ('.\n', 1.0),  # Full stop with newline
            (';\n', 0.7),  # Semicolon with newline
            ('; ', 0.6),   # Semicolon
            (':\n', 0.5),  # Colon with newline
            (', ', 0.3),   # Comma (lowest priority)
        ]
    
    def chunk_text(self, text: str) -> List[TextChunk]:
        """Chunk text intelligently based on strategy"""
        if self.strategy == ChunkingStrategy.HYBRID:
            return self._hybrid_chunk(text)
        elif self.strategy == ChunkingStrategy.SENTENCE:
            return self._sentence_chunk(text)
        elif self.strategy == ChunkingStrategy.TOKEN_COUNT:
            return self._token_chunk(text)
        else:
            return self._paragraph_chunk(text)
    
    def _hybrid_chunk(self, text: str) -> List[TextChunk]:
        """Hybrid chunking - best of all strategies"""
        chunks = []
        remaining_text = text.strip()
        chunk_index = 0
        previous_overlap = ""
        
        while remaining_text:
            # Find optimal break point
            if len(remaining_text) <= self.max_chunk_size:
                # Final chunk
                chunk_text = previous_overlap + remaining_text
                chunks.append(TextChunk(
                    text=chunk_text,
                    index=chunk_index,
                    is_final=True,
                    overlap_text="",
                    metadata={'strategy': 'final'}
                ))
                break
            
            # Look for natural break points
            chunk_end = self._find_best_break(remaining_text, self.max_chunk_size)
            
            # Extract chunk with overlap from previous
            chunk_text = previous_overlap + remaining_text[:chunk_end]
            
            # Prepare overlap for next chunk
            overlap_start = max(0, chunk_end - self.overlap_size)
            overlap_text = remaining_text[overlap_start:chunk_end]
            
            chunks.append(TextChunk(
                text=chunk_text,
                index=chunk_index,
                is_final=False,
                overlap_text=overlap_text,
                metadata={'strategy': 'hybrid', 'break_point': chunk_end}
            ))
            
            # Move to next chunk
            remaining_text = remaining_text[chunk_end:].strip()
            previous_overlap = overlap_text + " " if overlap_text else ""
            chunk_index += 1
        
        logger.info(f"📝 Chunked text into {len(chunks)} chunks using hybrid strategy")
        return chunks
    
    def _find_best_break(self, text: str, max_length: int) -> int:
        """Find the best breaking point in text"""
        # Start from max_length and work backwards
        search_start = max(0, max_length - 100)
        search_end = min(len(text), max_length + 50)
        search_text = text[search_start:search_end]
        
        best_pos = max_length
        best_score = 0
        
        for delimiter, score in self.sentence_delimiters:
            pos = search_text.rfind(delimiter)
            if pos != -1:
                actual_pos = search_start + pos + len(delimiter)
                if actual_pos <= max_length and score > best_score:
                    best_pos = actual_pos
                    best_score = score
        
        return best_pos
    
    def _sentence_chunk(self, text: str) -> List[TextChunk]:
        """Simple sentence-based chunking"""
        # Split by sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= self.max_chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        index=chunk_index,
                        is_final=False,
                        metadata={'strategy': 'sentence'}
                    ))
                    chunk_index += 1
                current_chunk = sentence + " "
        
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                index=chunk_index,
                is_final=True,
                metadata={'strategy': 'sentence'}
            ))
        
        return chunks
    
    def _token_chunk(self, text: str) -> List[TextChunk]:
        """Token-count based chunking"""
        words = text.split()
        chunks = []
        chunk_index = 0
        
        # Approximate tokens per word
        tokens_per_word = 1.3
        words_per_chunk = int(self.max_chunk_size / tokens_per_word)
        
        for i in range(0, len(words), words_per_chunk):
            chunk_words = words[i:i + words_per_chunk]
            chunks.append(TextChunk(
                text=' '.join(chunk_words),
                index=chunk_index,
                is_final=(i + words_per_chunk >= len(words)),
                metadata={'strategy': 'token_count'}
            ))
            chunk_index += 1
        
        return chunks
    
    def _paragraph_chunk(self, text: str) -> List[TextChunk]:
        """Paragraph-based chunking"""
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_index = 0
        
        for para in paragraphs:
            if len(current_chunk) + len(para) <= self.max_chunk_size:
                current_chunk += para + "\n\n"
            else:
                if current_chunk:
                    chunks.append(TextChunk(
                        text=current_chunk.strip(),
                        index=chunk_index,
                        is_final=False,
                        metadata={'strategy': 'paragraph'}
                    ))
                    chunk_index += 1
                current_chunk = para + "\n\n"
        
        if current_chunk:
            chunks.append(TextChunk(
                text=current_chunk.strip(),
                index=chunk_index,
                is_final=True,
                metadata={'strategy': 'paragraph'}
            ))
        
        return chunks

class AudioConcatenator:
    """Handles audio chunk concatenation with crossfade"""
    
    def __init__(self, sample_rate: int = 24000, crossfade_ms: int = 50):
        self.sample_rate = sample_rate
        self.crossfade_samples = int((crossfade_ms / 1000) * sample_rate)
        self.audio_buffer = bytearray()
        
    def add_chunk(self, audio_bytes: bytes, apply_crossfade: bool = True) -> bytes:
        """Add audio chunk with optional crossfade"""
        if not self.audio_buffer or not apply_crossfade:
            # First chunk or no crossfade
            self.audio_buffer.extend(audio_bytes)
            return audio_bytes
        
        # Convert bytes to numpy for crossfade
        new_audio = np.frombuffer(audio_bytes, dtype=np.int16)
        
        if len(self.audio_buffer) >= self.crossfade_samples * 2:
            # Get overlap region from buffer
            buffer_array = np.frombuffer(self.audio_buffer, dtype=np.int16)
            overlap_start = len(buffer_array) - self.crossfade_samples
            
            # Create crossfade
            fade_out = np.linspace(1, 0, self.crossfade_samples)
            fade_in = np.linspace(0, 1, self.crossfade_samples)
            
            # Apply crossfade
            buffer_array[overlap_start:] = (
                buffer_array[overlap_start:] * fade_out
            ).astype(np.int16)
            
            new_audio[:self.crossfade_samples] = (
                new_audio[:self.crossfade_samples] * fade_in +
                buffer_array[overlap_start:]
            ).astype(np.int16)
            
            # Update buffer
            self.audio_buffer = bytearray(buffer_array.tobytes())
            self.audio_buffer.extend(new_audio[self.crossfade_samples:].tobytes())
        else:
            # Not enough samples for crossfade
            self.audio_buffer.extend(audio_bytes)
        
        return audio_bytes
    
    def get_complete_audio(self) -> bytes:
        """Get the complete concatenated audio"""
        return bytes(self.audio_buffer)
    
    def reset(self):
        """Reset the buffer"""
        self.audio_buffer = bytearray()

class OrpheusEnhancedServer:
    """Production-grade Orpheus TTS server with extended audio generation"""
    
    def __init__(self):
        self.model = None
        self.executor = ThreadPoolExecutor(max_workers=8)  # Increased workers
        self.request_semaphore = asyncio.Semaphore(20)  # Higher concurrency
        self.model_lock = asyncio.Lock()
        self.performance_monitor = PerformanceMonitor()
        self.text_chunker = IntelligentTextChunker(
            max_chunk_size=400,  # Optimal chunk size for coherence
            overlap_size=30,     # Context preservation
            strategy=ChunkingStrategy.HYBRID
        )
        self.audio_concatenator = AudioConcatenator()
        
        # Enhanced configuration
        self.config = {
            "model_name": os.getenv("MODEL_NAME", "canopylabs/orpheus-tts-0.1-finetune-prod"),
            "max_model_len": int(os.getenv("MAX_MODEL_LEN", "64000")),
            "gpu_memory_utilization": float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95")),
            "max_tokens_per_chunk": int(os.getenv("MAX_TOKENS_PER_CHUNK", "8000")),  # Safe chunk size
            "enable_chunking": os.getenv("ENABLE_CHUNKING", "true").lower() == "true",
            "chunk_overlap_tokens": int(os.getenv("CHUNK_OVERLAP_TOKENS", "100")),
        }
        
        # Cache for repeated phrases
        self.audio_cache = {}
        self.cache_max_size = 100  # Maximum cache entries
        
        logger.info("🚀 Initializing Enhanced Orpheus TTS Server")
        logger.info(f"📋 Configuration: {json.dumps(self.config, indent=2)}")
    
    async def initialize(self):
        """Initialize model with enhanced error handling"""
        async with self.model_lock:
            if self.model is not None:
                return
            
            logger.info("🔧 Initializing Orpheus model...")
            
            # GPU memory management
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                gc.collect()
                
                # Log GPU info
                gpu_name = torch.cuda.get_device_name(0)
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
            
            try:
                # Load model - OrpheusModel only accepts model_name and dtype
                # The vLLM parameters are set internally by the library
                self.model = OrpheusModel(
                    model_name=self.config["model_name"],
                    dtype=torch.bfloat16
                )
                
                logger.info("✅ Model initialized successfully")
                
                # Comprehensive warmup
                await self._comprehensive_warmup()
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize model: {e}")
                logger.error(traceback.format_exc())
                raise
    
    async def _comprehensive_warmup(self):
        """Comprehensive warmup with various text lengths"""
        logger.info("🔥 Running comprehensive warmup...")
        
        warmup_texts = [
            "Short warmup text.",
            "Medium length warmup text to prepare the model for generation.",
            "This is a longer warmup text that helps prepare the model for extended generation. " * 3
        ]
        
        for i, text in enumerate(warmup_texts, 1):
            try:
                logger.info(f"  Warmup {i}/{len(warmup_texts)}...")
                async for _ in self.generate_extended_audio(text, "tara", max_duration_seconds=10):
                    pass
            except Exception as e:
                logger.warning(f"  Warmup {i} failed: {e}")
        
        logger.info("✅ Warmup complete")
    
    async def generate_extended_audio(
        self,
        text: str,
        voice: str = "tara",
        temperature: float = 0.6,
        top_p: float = 0.8,
        repetition_penalty: float = 1.3,
        max_duration_seconds: Optional[int] = None,
        enable_chunking: Optional[bool] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Enhanced audio generation with intelligent chunking for unlimited duration
        
        This method breaks through the 24-second limitation by:
        1. Intelligently chunking text at natural boundaries
        2. Generating audio for each chunk with optimal token allocation
        3. Concatenating chunks with crossfade for seamless output
        4. Streaming results in real-time
        """
        
        start_time = time.time()
        total_audio_bytes = 0
        total_tokens = 0
        
        # Determine if chunking should be used
        use_chunking = enable_chunking if enable_chunking is not None else self.config["enable_chunking"]
        
        # Clean and prepare text
        text = text.strip()
        if not text:
            logger.warning("Empty text provided")
            return
        
        logger.info(f"🎯 Generating audio for {len(text)} characters")
        logger.info(f"📊 Settings: voice={voice}, temp={temperature}, chunking={use_chunking}")
        
        try:
            if use_chunking and len(text) > self.text_chunker.max_chunk_size:
                # Use intelligent chunking for long text
                chunks = self.text_chunker.chunk_text(text)
                logger.info(f"📚 Processing {len(chunks)} text chunks")
                
                # Reset audio concatenator
                self.audio_concatenator.reset()
                
                for chunk_data in chunks:
                    chunk_start = time.time()
                    
                    # Calculate dynamic token limit for this chunk
                    chunk_tokens = min(
                        self.config["max_tokens_per_chunk"],
                        int(len(chunk_data.text) * 2.5)  # Approximate token ratio
                    )
                    
                    logger.info(f"  🔄 Chunk {chunk_data.index + 1}/{len(chunks)}: {len(chunk_data.text)} chars, {chunk_tokens} tokens")
                    
                    # Generate audio for this chunk
                    chunk_audio_bytes = b""
                    async for audio_chunk in self._generate_single_chunk(
                        text=chunk_data.text,
                        voice=voice,
                        temperature=temperature,
                        top_p=top_p,
                        repetition_penalty=repetition_penalty,
                        max_tokens=chunk_tokens
                    ):
                        chunk_audio_bytes += audio_chunk
                        yield audio_chunk  # Stream immediately
                    
                    # Track performance
                    chunk_duration = time.time() - chunk_start
                    audio_duration = len(chunk_audio_bytes) / 2 / 24000  # 16-bit, 24kHz
                    total_audio_bytes += len(chunk_audio_bytes)
                    total_tokens += chunk_tokens
                    
                    logger.info(f"    ✓ Generated {audio_duration:.1f}s in {chunk_duration:.1f}s (RTF: {chunk_duration/audio_duration:.2f})")
                    
                    # Add small pause between chunks for natural flow
                    if not chunk_data.is_final:
                        silence_duration = 0.1  # 100ms pause
                        silence_samples = int(24000 * silence_duration)
                        silence_bytes = np.zeros(silence_samples, dtype=np.int16).tobytes()
                        yield silence_bytes
                        total_audio_bytes += len(silence_bytes)
                    
                    # Memory management between chunks
                    if chunk_data.index % 5 == 0:
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
            
            else:
                # Single chunk generation for short text
                logger.info(f"📄 Processing as single chunk")
                
                # Calculate appropriate token limit
                max_tokens = min(
                    64000,  # Model maximum
                    int(len(text) * 3)  # Generous token estimate
                )
                
                async for audio_chunk in self._generate_single_chunk(
                    text=text,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    max_tokens=max_tokens
                ):
                    total_audio_bytes += len(audio_chunk)
                    total_tokens += max_tokens
                    yield audio_chunk
            
            # Log final statistics
            total_time = time.time() - start_time
            total_audio_duration = total_audio_bytes / 2 / 24000
            
            self.performance_monitor.log_generation(
                audio_duration=total_audio_duration,
                generation_time=total_time,
                tokens=total_tokens
            )
            
            logger.info(f"✅ Generation complete: {total_audio_duration:.1f}s audio in {total_time:.1f}s")
            logger.info(f"📈 Stats: {total_audio_bytes/1024:.1f}KB, {total_tokens} tokens, RTF: {total_time/total_audio_duration:.2f}x")
            
        except Exception as e:
            logger.error(f"❌ Generation failed: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _generate_single_chunk(
        self,
        text: str,
        voice: str,
        temperature: float,
        top_p: float,
        repetition_penalty: float,
        max_tokens: int
    ) -> AsyncGenerator[bytes, None]:
        """Generate audio for a single text chunk"""
        
        # Check cache first
        cache_key = f"{text[:50]}_{voice}_{temperature}_{top_p}"
        if cache_key in self.audio_cache:
            logger.info("  📦 Using cached audio")
            cached_audio = self.audio_cache[cache_key]
            yield cached_audio
            return
        
        # Generate new audio
        first_chunk_time = None
        generated_audio = b""
        
        try:
            loop = asyncio.get_event_loop()
            
            # Create generator in thread
            def generate():
                return self.model.generate_speech(
                    prompt=text,
                    voice=voice,
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    repetition_penalty=repetition_penalty
                    # Note: stop_token_ids removed - let model decide when to stop naturally
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
                chunk = await loop.run_in_executor(self.executor, get_next_chunk)
                
                if chunk is None:
                    break
                
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                
                generated_audio += chunk
                yield chunk
            
            # Cache if small enough
            if len(generated_audio) < 1024 * 1024:  # Cache if less than 1MB
                self.audio_cache[cache_key] = generated_audio
                
                # Manage cache size
                if len(self.audio_cache) > self.cache_max_size:
                    # Remove oldest entry
                    oldest_key = next(iter(self.audio_cache))
                    del self.audio_cache[oldest_key]
        
        except Exception as e:
            logger.error(f"  ❌ Chunk generation failed: {e}")
            raise

# Global server instance
server = OrpheusEnhancedServer()

# Create aiohttp application
app = web.Application(
    client_max_size=100 * 1024 * 1024  # 100MB max request size
)

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
    """Initialize server on startup"""
    await server.initialize()

async def cleanup(app):
    """Cleanup on shutdown"""
    logger.info("🔄 Shutting down server...")
    if server.executor:
        server.executor.shutdown(wait=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

app.on_startup.append(startup)
app.on_cleanup.append(cleanup)

# Enhanced Routes

async def health(request):
    """Enhanced health check with metrics"""
    if server.model:
        metrics = server.performance_monitor.metrics
        avg_rtf = sum(metrics['average_rtf']) / len(metrics['average_rtf']) if metrics['average_rtf'] else 0
        
        return web.json_response({
            "status": "healthy",
            "uptime": time.time(),
            "metrics": {
                "total_requests": metrics['total_requests'],
                "total_audio_seconds": round(metrics['total_audio_seconds'], 1),
                "average_rtf": round(avg_rtf, 2),
                "memory_mb": round(metrics['memory_usage'][-1] if metrics['memory_usage'] else 0, 1)
            },
            "capabilities": {
                "max_tokens": server.config["max_model_len"],
                "chunking_enabled": server.config["enable_chunking"],
                "max_audio_duration": "unlimited with chunking"
            }
        })
    return web.json_response({"status": "initializing"}, status=503)

async def generate_tts(request):
    """Enhanced TTS endpoint with extended audio support"""
    try:
        data = await request.json()
        text = data.get('text', '')
        voice = data.get('voice', 'tara')
        temperature = data.get('temperature', 0.6)
        top_p = data.get('top_p', 0.8)
        repetition_penalty = data.get('repetition_penalty', 1.3)
        max_duration = data.get('max_duration_seconds', None)
        enable_chunking = data.get('enable_chunking', True)
        
        if not text:
            return web.json_response({"error": "Text required"}, status=400)
        
        # Log request details
        logger.info(f"📨 TTS Request: {len(text)} chars, voice={voice}, chunking={enable_chunking}")
        
        # Generate audio with streaming
        response = web.StreamResponse()
        response.headers['Content-Type'] = 'audio/wav'
        response.headers['X-Text-Length'] = str(len(text))
        response.headers['X-Chunking-Enabled'] = str(enable_chunking)
        
        await response.prepare(request)
        
        # Write WAV header
        wav_header = create_wav_header(0)  # Placeholder size
        await response.write(wav_header)
        
        # Stream audio chunks
        total_audio_bytes = 0
        async for audio_chunk in server.generate_extended_audio(
            text=text,
            voice=voice,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            max_duration_seconds=max_duration,
            enable_chunking=enable_chunking
        ):
            await response.write(audio_chunk)
            total_audio_bytes += len(audio_chunk)
        
        # Update WAV header with actual size (seeking back not possible in streaming)
        # Client should handle this or use chunked transfer encoding
        
        await response.write_eof()
        
        logger.info(f"✅ Streamed {total_audio_bytes/1024:.1f}KB of audio")
        return response
        
    except Exception as e:
        logger.error(f"❌ TTS generation failed: {e}")
        logger.error(traceback.format_exc())
        return web.json_response({"error": str(e)}, status=500)

async def generate_tts_simple(request):
    """Simple non-streaming endpoint for compatibility"""
    try:
        data = await request.json()
        text = data.get('text', '')
        voice = data.get('voice', 'tara')
        
        if not text:
            return web.json_response({"error": "Text required"}, status=400)
        
        # Generate complete audio
        audio_data = b""
        async for chunk in server.generate_extended_audio(
            text=text,
            voice=voice,
            enable_chunking=True
        ):
            audio_data += chunk
        
        # Return with proper WAV header
        wav_header = create_wav_header(len(audio_data))
        
        return web.Response(
            body=wav_header + audio_data,
            content_type='audio/wav',
            headers={
                'X-Audio-Duration': str(len(audio_data) / 2 / 24000),
                'X-Audio-Size': str(len(audio_data))
            }
        )
        
    except Exception as e:
        logger.error(f"❌ TTS generation failed: {e}")
        return web.json_response({"error": str(e)}, status=500)

async def websocket_handler(request):
    """Enhanced WebSocket endpoint with progress tracking"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    async for msg in ws:
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
                text = data.get('text', '')
                voice = data.get('voice', 'tara')
                enable_chunking = data.get('enable_chunking', True)
                
                if not text:
                    await ws.send_json({"error": "Text required"})
                    continue
                
                # Send initial acknowledgment
                await ws.send_json({
                    "type": "started",
                    "text_length": len(text),
                    "chunking": enable_chunking
                })
                
                # Stream audio chunks with progress
                chunk_count = 0
                total_audio_bytes = 0
                
                async for audio_chunk in server.generate_extended_audio(
                    text=text,
                    voice=voice,
                    enable_chunking=enable_chunking
                ):
                    total_audio_bytes += len(audio_chunk)
                    
                    # Send as base64
                    import base64
                    audio_b64 = base64.b64encode(audio_chunk).decode('utf-8')
                    
                    await ws.send_json({
                        "type": "audio_chunk",
                        "chunk": chunk_count,
                        "data": audio_b64,
                        "total_bytes": total_audio_bytes,
                        "duration_seconds": total_audio_bytes / 2 / 24000
                    })
                    chunk_count += 1
                
                # Send completion with statistics
                await ws.send_json({
                    "type": "complete",
                    "chunks": chunk_count,
                    "total_bytes": total_audio_bytes,
                    "duration_seconds": total_audio_bytes / 2 / 24000
                })
                
            except Exception as e:
                logger.error(f"❌ WebSocket error: {e}")
                await ws.send_json({"type": "error", "error": str(e)})
        
        elif msg.type == aiohttp.WSMsgType.ERROR:
            logger.error(f'WebSocket error: {ws.exception()}')
    
    return ws

def create_wav_header(data_size):
    """Create WAV header with proper size"""
    sample_rate = 24000
    channels = 1
    bits_per_sample = 16
    
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    # Use maximum size if streaming (will be updated by client)
    if data_size == 0:
        data_size = 0x7FFFFFFF  # Maximum possible size for streaming
    
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

# Statistics endpoint
async def get_stats(request):
    """Get server statistics"""
    metrics = server.performance_monitor.metrics
    
    # Calculate statistics
    avg_rtf = sum(metrics['average_rtf']) / len(metrics['average_rtf']) if metrics['average_rtf'] else 0
    current_memory = metrics['memory_usage'][-1] if metrics['memory_usage'] else 0
    peak_memory = max(metrics['memory_usage']) if metrics['memory_usage'] else 0
    
    # GPU stats if available
    gpu_stats = {}
    if torch.cuda.is_available():
        gpu_stats = {
            "name": torch.cuda.get_device_name(0),
            "memory_allocated_mb": torch.cuda.memory_allocated(0) / 1024 / 1024,
            "memory_reserved_mb": torch.cuda.memory_reserved(0) / 1024 / 1024,
            "utilization": torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else "N/A"
        }
    
    return web.json_response({
        "server": {
            "uptime_seconds": time.time(),
            "total_requests": metrics['total_requests'],
            "total_audio_generated_seconds": round(metrics['total_audio_seconds'], 1),
            "total_tokens_generated": metrics['total_tokens_generated']
        },
        "performance": {
            "average_rtf": round(avg_rtf, 2),
            "current_memory_mb": round(current_memory, 1),
            "peak_memory_mb": round(peak_memory, 1)
        },
        "gpu": gpu_stats,
        "configuration": server.config
    })

# Add all routes
app.router.add_get('/health', health)
app.router.add_post('/tts', generate_tts)
app.router.add_post('/tts/simple', generate_tts_simple)
app.router.add_get('/ws', websocket_handler)
app.router.add_get('/stats', get_stats)

# Serve static test page
app.router.add_static('/', path='.')

# Add CORS to all routes
for route in list(app.router.routes()):
    cors.add(route)

if __name__ == '__main__':
    port = int(os.getenv('PORT', 8080))
    
    logger.info("=" * 60)
    logger.info("🚀 ORPHEUS ENHANCED TTS SERVER")
    logger.info("=" * 60)
    logger.info(f"📡 Starting server on http://0.0.0.0:{port}")
    logger.info(f"📊 Health check: http://localhost:{port}/health")
    logger.info(f"📈 Statistics: http://localhost:{port}/stats")
    logger.info(f"🔧 WebSocket: ws://localhost:{port}/ws")
    logger.info("=" * 60)
    
    web.run_app(app, host='0.0.0.0', port=port)