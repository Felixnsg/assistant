"""
RealtimeTTS Server Script

This script runs on your GPU instance and provides a REST API for text-to-speech
conversion using CoquiEngine. It accepts text requests and returns audio data.

Run this on your GPU server with:
    python tts_server.py
"""

from flask import Flask, request, Response, jsonify
import io
import logging
import os
import tempfile
import time
import wave
import numpy as np
from RealtimeTTS.engines import CoquiEngine
from RealtimeTTS import TextToAudioStream

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Initialize CoquiEngine on startup
logger.info("Initializing CoquiEngine...")
engine = CoquiEngine(use_gpu=True)
logger.info(f"CoquiEngine initialized with languages: {engine.available_languages}")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify server is running"""
    return jsonify({
        "status": "healthy",
        "engine": "CoquiEngine",
        "available_languages": engine.available_languages
    })

@app.route('/tts', methods=['POST'])
def text_to_speech():
    """Convert text to speech and return the audio data"""
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Request must include 'text' field"}), 400
    
    text = request.json['text']
    language = request.json.get('language', 'en')
    logger.info(f"Received TTS request: {len(text)} characters, language: {language}")
    
    # Create a temporary WAV file
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
        temp_path = temp_file.name
    
    try:
        # Create audio stream
        stream = TextToAudioStream(engine=engine)
        
        # Feed text and generate audio
        stream.feed(text)
        
        # Process and save to WAV file
        logger.info(f"Generating audio for: '{text[:50]}...'")
        start_time = time.time()
        stream.play(output_wavfile=temp_path, muted=True)
        
        processing_time = time.time() - start_time
        logger.info(f"Audio generated in {processing_time:.2f} seconds")
        
        # Read the WAV file and return as response
        with open(temp_path, 'rb') as audio_file:
            audio_data = audio_file.read()
        
        return Response(
            audio_data,
            mimetype='audio/wav'
        )
    
    except Exception as e:
        logger.error(f"Error processing TTS request: {str(e)}")
        return jsonify({"error": str(e)}), 500
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.route('/tts-stream', methods=['POST'])
def text_to_speech_stream():
    """Stream audio data as it's being generated"""
    if not request.json or 'text' not in request.json:
        return jsonify({"error": "Request must include 'text' field"}), 400
    
    text = request.json['text']
    language = request.json.get('language', 'en')
    logger.info(f"Received streaming TTS request: {len(text)} characters, language: {language}")
    
    def generate():
        """Generator function to stream audio chunks"""
        # Process text in chunks for streaming
        stream = TextToAudioStream(engine=engine)
        stream.feed(text)
        
        # Create a temporary in-memory buffer
        buffer = io.BytesIO()
        
        # Configure WAV header
        sample_rate = 24000  # Coqui typically uses 24kHz
        num_channels = 1
        bytes_per_sample = 2  # 16-bit audio
        
        # Create a WAV writer for the buffer
        wav_writer = wave.open(buffer, 'wb')
        wav_writer.setnchannels(num_channels)
        wav_writer.setsampwidth(bytes_per_sample)
        wav_writer.setframerate(sample_rate)
        
        # Process audio in chunks
        def on_audio_chunk(chunk):
            # Convert chunk to bytes and write to WAV buffer
            if isinstance(chunk, np.ndarray):
                # Convert numpy array to bytes
                chunk_bytes = (chunk * 32767).astype(np.int16).tobytes()
                wav_writer.writeframes(chunk_bytes)
                
                # Yield the chunk as part of the stream response
                with io.BytesIO() as chunk_buffer:
                    chunk_wav = wave.open(chunk_buffer, 'wb')
                    chunk_wav.setnchannels(num_channels)
                    chunk_wav.setsampwidth(bytes_per_sample)
                    chunk_wav.setframerate(sample_rate)
                    chunk_wav.writeframes(chunk_bytes)
                    chunk_wav.close()
                    yield chunk_buffer.getvalue()
        
        # Start processing with callbacks for chunks
        stream.play(muted=True, on_audio_chunk=on_audio_chunk)
        
        # Close the WAV writer
        wav_writer.close()
    
    # Return a streaming response with audio/wav MIME type
    return Response(
        generate(),
        mimetype='audio/wav'
    )

if __name__ == '__main__':
    # Run the Flask app
    # Note: Use 0.0.0.0 to make the server accessible from other machines
    app.run(host='0.0.0.0', port=8080, debug=False)