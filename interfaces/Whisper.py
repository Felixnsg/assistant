# File: Whisper.py
"""
Flask application providing a simple API endpoint for audio transcription
using the OpenAI Whisper model.
"""

from flask import Flask, request, jsonify, send_from_directory # Added send_from_directory for potential future use
import whisper
import tempfile
import os
import logging 
import torch 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "large") # Default to large
MODEL_PATH = os.getenv("WHISPER_MODEL_PATH", None) # Optional path for custom models/cache
whisper_model = None
try:
    logging.info(f"Loading Whisper model '{MODEL_SIZE}'...")
    # Check for GPU availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    whisper_model = whisper.load_model(MODEL_SIZE, download_root=MODEL_PATH, device=device)
    logging.info(f"Whisper model '{MODEL_SIZE}' loaded successfully.")
except Exception as e:
    logging.error(f"FATAL: Failed to load Whisper model '{MODEL_SIZE}': {e}", exc_info=True)
    # sys.exit(1)
    whisper_model = None # Ensure it's None if loading failed

@app.route('/transcribe', methods=['POST'])
def transcription():
    """
    Handle POST requests to transcribe an audio file.

    Expects a multipart/form-data request with a file field named 'file'.

    Returns:
        JSON response with transcription result or error details.
    """
    if whisper_model is None:
        logging.error("Transcription request received, but Whisper model is not loaded.")
        return jsonify({
            "Result": "Failure",
            "Text": "Whisper model not available on server."
        }), 503 # Service Unavailable

    if "file" not in request.files:
        logging.warning("Transcription request received with no 'file' part.")
        return jsonify({
            "Result": "Failure",
            "Text": "No file part in the request."
        }), 400 # Bad Request

    file = request.files["file"]

    if file.filename == '':
        logging.warning("Transcription request received with empty filename.")
        return jsonify({
            "Result": "Failure",
            "Text": "No selected file."
        }), 400 # Bad Request

    audio_filename = None     # Initialize
    temp_dir = tempfile.gettempdir() # Get system temp dir

    try:
        # Create a temporary file with a safe name
        # Suffix helps Whisper identify file type, though it mainly uses content
        suffix = os.path.splitext(file.filename)[1] or ".wav" # Use original suffix or default
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=temp_dir) as temp_fp:
            file.save(temp_fp.name)
            audio_filename = temp_fp.name
        logging.info(f"Received file '{file.filename}', saved temporarily as '{audio_filename}'")

        logging.info(f"Starting transcription for '{audio_filename}'...")
        # Add options like language detection if needed: result = model.transcribe(audio_filename, language='en')
        result = whisper_model.transcribe(audio_filename)
        transcription = result["text"]
        logging.info(f"Transcription complete. Length: {len(transcription)} chars.")

        # --- REFACTOR: Always return success structure ---
        return jsonify({
            "Result": "Success",
            "Text": transcription
        }), 200

    except whisper.errors.WhisperError as e: # Catch specific Whisper errors if available
        logging.error(f"Whisper transcription error for file '{audio_filename}': {e}", exc_info=True)
        return jsonify({
            "Result": "Failure",
            "Text": f"Transcription error: {e}"
        }), 500 # Internal Server Error
    except FileNotFoundError:
         logging.error(f"Error: Temporary file '{audio_filename}' not found during transcription.", exc_info=True)
         return jsonify({"Result": "Failure", "Text": "Temporary file error."}), 500
    except Exception as e:
        logging.error(f"Unexpected error during transcription of file '{audio_filename}': {e}", exc_info=True)
        return jsonify({
            "Result": "Failure",
            "Text": f"An unexpected server error occurred: {e}"
        }), 500 # Internal Server Error
    finally:
        if audio_filename and os.path.exists(audio_filename):
            try:
                os.unlink(audio_filename)
                logging.info(f"Temporary file '{audio_filename}' deleted.")
            except OSError as e:
                logging.error(f"Error deleting temporary file '{audio_filename}': {e}", exc_info=True)

# simple health check endpoint ---
@app.route('/health', methods=['GET'])
def health_check():
    """Basic health check endpoint."""
    if whisper_model:
        return jsonify({"status": "ok", "model_loaded": True, "model_size": MODEL_SIZE}), 200
    else:
        return jsonify({"status": "error", "model_loaded": False, "message": "Whisper model failed to load."}), 503

if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '0.0.0.0')
    port = int(os.getenv('FLASK_PORT', 5001))
    debug_mode = os.getenv('FLASK_DEBUG', 'False').lower() in ('true', '1', 't')

    logging.info(f"Starting Whisper API server on {host}:{port} (Debug: {debug_mode})")
    # Use waitress or gunicorn for production instead of app.run(debug=True)
    app.run(host=host, port=port, debug=debug_mode)