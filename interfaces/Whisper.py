from flask import Flask, request, jsonify
import whisper
import tempfile
import os

app = Flask(__name__)

# Load whisper model once at startup (efficient)
model = whisper.load_model("large")  # or whatever size you're using

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
        file.save(temp.name)
        temp_filename = temp.name
    
    try:
        # Transcribe with Whisper
        result = model.transcribe(temp_filename)
        transcription = result["text"].strip()
        
        # Clean up
        os.unlink(temp_filename)
        
        return jsonify({
            "success": True,
            "transcription": transcription
        })
    except Exception as e:
        # Clean up on error
        if os.path.exists(temp_filename):
            os.unlink(temp_filename)
        
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)