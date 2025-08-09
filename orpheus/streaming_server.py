#!/usr/bin/env python3
"""
Orpheus TTS Streaming Server
Real-time audio streaming over HTTP with Flask
"""

from flask import Flask, Response, request, jsonify
import struct
import time
import torch
from orpheus_tts import OrpheusModel
import threading
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model instance
model = None
model_lock = threading.Lock()

def create_wav_header(sample_rate=24000, bits_per_sample=16, channels=1):
    """Create WAV file header for streaming"""
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    
    # Use placeholder for data size (streaming)
    data_size = 0xFFFFFFFF  # Maximum size for streaming
    
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        36 + data_size,
        b'WAVE',
        b'fmt ',
        16,                   # fmt chunk size
        1,                    # PCM format
        channels,
        sample_rate,
        byte_rate,
        block_align,
        bits_per_sample,
        b'data',
        data_size
    )
    return header

def initialize_model():
    """Initialize the Orpheus model"""
    global model
    with model_lock:
        if model is None:
            logger.info("Initializing Orpheus TTS model...")
            try:
                model = OrpheusModel(
                    model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
                    dtype=torch.bfloat16,
                    max_model_len=2048
                )
                logger.info("✅ Model loaded successfully")
            except Exception as e:
                logger.error(f"❌ Failed to load model: {e}")
                raise

@app.route('/')
def index():
    """Root endpoint with API documentation"""
    return """
    <h1>Orpheus TTS Streaming Server</h1>
    <h2>Endpoints:</h2>
    <ul>
        <li><b>GET /tts</b> - Stream TTS audio</li>
        <li><b>GET /tts_chunked</b> - Stream TTS with chunk timing info</li>
        <li><b>GET /voices</b> - List available voices</li>
        <li><b>GET /health</b> - Server health check</li>
    </ul>
    <h3>TTS Parameters:</h3>
    <ul>
        <li>text (required): Text to synthesize</li>
        <li>voice (optional): Voice name (default: tara)</li>
        <li>temperature (optional): Generation temperature (default: 0.6)</li>
        <li>top_p (optional): Top-p sampling (default: 0.8)</li>
        <li>repetition_penalty (optional): Repetition penalty (default: 1.3)</li>
    </ul>
    <h3>Example:</h3>
    <code>curl "http://localhost:8080/tts?text=Hello%20world&voice=leo" > output.wav</code>
    """

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "cuda_available": torch.cuda.is_available()
    })

@app.route('/voices', methods=['GET'])
def voices():
    """List available voices"""
    return jsonify({
        "voices": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    })

@app.route('/tts', methods=['GET', 'POST'])
def tts():
    """Main TTS streaming endpoint"""
    # Get parameters
    if request.method == 'POST':
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'tara')
        temperature = data.get('temperature', 0.6)
        top_p = data.get('top_p', 0.8)
        repetition_penalty = data.get('repetition_penalty', 1.3)
    else:
        text = request.args.get('text', '')
        voice = request.args.get('voice', 'tara')
        temperature = float(request.args.get('temperature', '0.6'))
        top_p = float(request.args.get('top_p', '0.8'))
        repetition_penalty = float(request.args.get('repetition_penalty', '1.3'))
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    logger.info(f"TTS request - Voice: {voice}, Text length: {len(text)} chars")
    
    def generate_audio_stream():
        """Generator function for streaming audio"""
        try:
            # Send WAV header first
            yield create_wav_header()
            
            start_time = time.time()
            first_chunk_sent = False
            total_bytes = 0
            chunk_count = 0
            
            # Generate speech
            syn_tokens = model.generate_speech(
                prompt=text,
                voice=voice,
                temperature=temperature,
                top_p=top_p,
                max_tokens=2000,
                repetition_penalty=repetition_penalty,
                stop_token_ids=[128258]
            )
            
            # Stream audio chunks
            for audio_chunk in syn_tokens:
                if not first_chunk_sent:
                    ttfb = (time.time() - start_time) * 1000
                    logger.info(f"⏱️ Time to first byte: {ttfb:.1f}ms")
                    first_chunk_sent = True
                
                chunk_count += 1
                total_bytes += len(audio_chunk)
                yield audio_chunk
            
            # Log final statistics
            total_time = time.time() - start_time
            logger.info(f"✅ Streamed {chunk_count} chunks, {total_bytes} bytes in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Error generating audio: {e}")
            # Send silence on error to maintain stream
            yield b'\x00' * 1024
    
    return Response(generate_audio_stream(), mimetype='audio/wav')

@app.route('/tts_chunked', methods=['GET', 'POST'])
def tts_chunked():
    """TTS endpoint with chunk timing information (for debugging)"""
    # Get parameters (same as /tts)
    if request.method == 'POST':
        data = request.get_json()
        text = data.get('text', '')
        voice = data.get('voice', 'tara')
    else:
        text = request.args.get('text', '')
        voice = request.args.get('voice', 'tara')
    
    if not text:
        return jsonify({"error": "No text provided"}), 400
    
    def generate_chunked_stream():
        """Generator with timing markers"""
        yield b'--CHUNK_BOUNDARY\r\n'
        yield b'Content-Type: audio/wav\r\n\r\n'
        yield create_wav_header()
        yield b'\r\n--CHUNK_BOUNDARY\r\n'
        
        start_time = time.time()
        chunk_id = 0
        
        syn_tokens = model.generate_speech(
            prompt=text,
            voice=voice,
            temperature=0.6,
            top_p=0.8,
            max_tokens=2000,
            repetition_penalty=1.3,
            stop_token_ids=[128258]
        )
        
        for audio_chunk in syn_tokens:
            elapsed = (time.time() - start_time) * 1000
            
            # Send timing metadata
            yield f'X-Chunk-ID: {chunk_id}\r\n'.encode()
            yield f'X-Chunk-Time: {elapsed:.1f}ms\r\n'.encode()
            yield f'X-Chunk-Size: {len(audio_chunk)}\r\n'.encode()
            yield b'Content-Type: audio/wav\r\n\r\n'
            yield audio_chunk
            yield b'\r\n--CHUNK_BOUNDARY\r\n'
            
            chunk_id += 1
        
        yield b'--CHUNK_BOUNDARY--\r\n'
    
    return Response(
        generate_chunked_stream(),
        mimetype='multipart/x-mixed-replace; boundary=CHUNK_BOUNDARY'
    )

@app.before_request
def before_request():
    """Ensure model is loaded before processing requests"""
    if request.endpoint in ['tts', 'tts_chunked']:
        initialize_model()

if __name__ == '__main__':
    print("=" * 60)
    print("ORPHEUS TTS STREAMING SERVER")
    print("=" * 60)
    print("\n📡 Starting server on http://0.0.0.0:8080")
    print("📖 Visit http://localhost:8080 for API documentation\n")
    
    # Initialize model on startup (optional)
    print("🔄 Loading model (this may take a moment)...")
    try:
        initialize_model()
        print("✅ Model ready!\n")
    except Exception as e:
        print(f"⚠️ Model will be loaded on first request: {e}\n")
    
    # Run Flask server
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)