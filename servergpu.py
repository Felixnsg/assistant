"""
RealtimeTTS Server Script (Fixed)

This script runs on your GPU instance and provides a REST API for text-to-speech
conversion using CoquiEngine. It accepts text requests and returns audio data.

Run this on your GPU server with:
    python servergpu.py
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

# Initialize CoquiEngine on startup - without use_gpu parameter
logger.info("Initializing CoquiEngine...")
engine = CoquiEngine()  # CoquiEngine should use GPU by default if available
logger.info("CoquiEngine initialized successfully")

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint to verify server is running"""
    return jsonify({
        "status": "healthy",
        "engine": "CoquiEngine"
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

if __name__ == '__main__':
    # Run the Flask app
    # Note: Use 0.0.0.0 to make the server accessible from other machines
    app.run(host='0.0.0.0', port=8080, debug=False)