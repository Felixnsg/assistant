# server.py
import os
import io
import base64
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Check for GPU availability
if torch.cuda.is_available():
    logger.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    logger.warning("No GPU found, falling back to CPU")
    device = "cpu"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Import RealtimeTTS after checking for GPU to ensure proper device assignment
from RealtimeTTS import TextToAudioStream
# Choose the appropriate engine based on GPU capability
# CoquiEngine is good for high-quality local synthesis with GPU acceleration
from RealtimeTTS import CoquiEngine, PiperEngine, SystemEngine

# Initialize TTS engine with GPU support
try:
    logger.info("Initializing CoquiEngine (primary choice for GPU)...")
    engine = CoquiEngine(device=device)
    logger.info("CoquiEngine initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize CoquiEngine: {e}")
    try:
        logger.info("Falling back to PiperEngine...")
        # Default to a standard Piper voice model
        engine = PiperEngine()
        logger.info("PiperEngine initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize PiperEngine: {e}")
        logger.info("Falling back to SystemEngine...")
        engine = SystemEngine()
        logger.info("SystemEngine initialized successfully")

@app.route('/tts', methods=['POST'])
def text_to_speech():
    try:
        data = request.json
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text']
        voice = data.get('voice', None)  # Optional voice parameter
        
        logger.info(f"Received TTS request: {text[:50]}...")
        
        # Configure engine if voice is specified
        if voice and hasattr(engine, 'set_voice'):
            engine.set_voice(voice)
        
        # Create a stream to a file instead of playing
        stream = TextToAudioStream(engine, muted=True)
        
        # Process the text
        stream.feed(text)
        
        # Generate audio and save to a BytesIO object
        memory_file = io.BytesIO()
        stream.play(output_wavfile=memory_file, reset_generated_text=True)
        memory_file.seek(0)
        
        # Return the audio file
        return send_file(
            memory_file,
            mimetype='audio/wav',
            as_attachment=True,
            download_name='tts_output.wav'
        )
    
    except Exception as e:
        logger.error(f"Error in text_to_speech: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/voices', methods=['GET'])
def list_voices():
    try:
        if hasattr(engine, 'list_voices'):
            voices = engine.list_voices()
            return jsonify({'voices': voices})
        else:
            return jsonify({'voices': ["default"]})
    except Exception as e:
        logger.error(f"Error in list_voices: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'ok', 'device': device})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)