# server.py
import os
import io
import base64
import logging
import multiprocessing

# IMPORTANT: Set the multiprocessing start method to 'spawn' before any imports that use torch
# This fixes the "Cannot re-initialize CUDA in forked subprocess" error
multiprocessing.set_start_method('spawn', force=True)

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import torch

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

# Try different TTS engines in order of preference
def init_tts_engine():
    # Initialize with a simple engine that always works
    from RealtimeTTS import SystemEngine
    current_engine = SystemEngine()
    engine_name = "SystemEngine"
    
    # Try to load better engines if possible
    try:
        from RealtimeTTS import GTTSEngine
        logger.info("Initializing GTTSEngine...")
        current_engine = GTTSEngine()
        engine_name = "GTTSEngine"
        logger.info("GTTSEngine initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize GTTSEngine: {e}")

    # Only try CoquiEngine if we have CUDA
    if device == "cuda":
        try:
            from RealtimeTTS import CoquiEngine
            logger.info("Initializing CoquiEngine with CUDA...")
            current_engine = CoquiEngine(device=device)
            engine_name = "CoquiEngine (GPU accelerated)"
            logger.info("CoquiEngine initialized successfully")
        except Exception as e:
            logger.warning(f"Failed to initialize CoquiEngine: {e}")

    try:
        from RealtimeTTS import PiperEngine
        logger.info("Initializing PiperEngine...")
        current_engine = PiperEngine()
        engine_name = "PiperEngine"
        logger.info("PiperEngine initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize PiperEngine: {e}")
            
    return current_engine, engine_name

# Initialize the TTS engine
engine, engine_name = init_tts_engine()
logger.info(f"Using {engine_name} for text-to-speech")

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
        
        # Import here to avoid circular imports
        from RealtimeTTS import TextToAudioStream
        
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
        
        # Fallback to gTTS if RealtimeTTS fails
        try:
            from gtts import gTTS
            memory_file = io.BytesIO()
            tts = gTTS(text=text, lang='en')
            tts.write_to_fp(memory_file)
            memory_file.seek(0)
            
            logger.info("Falling back to gTTS for this request")
            
            return send_file(
                memory_file,
                mimetype='audio/mp3',
                as_attachment=True,
                download_name='tts_output.mp3'
            )
        except Exception as fallback_error:
            logger.error(f"Fallback also failed: {fallback_error}")
            return jsonify({'error': str(e), 'fallback_error': str(fallback_error)}), 500

@app.route('/voices', methods=['GET'])
def list_voices():
    try:
        if hasattr(engine, 'list_voices'):
            voices = engine.list_voices()
            return jsonify({'voices': voices, 'engine': engine_name})
        else:
            return jsonify({'voices': ["default"], 'engine': engine_name})
    except Exception as e:
        logger.error(f"Error in list_voices: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'ok', 
        'device': device, 
        'engine': engine_name,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 6565))
    app.run(host='0.0.0.0', port=port)