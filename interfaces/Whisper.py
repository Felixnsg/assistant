from flask import Flask, request, jsonify
import whisper
import tempfile
import os


app = Flask(__name__)

model = whisper.load_model("large")

@app.route('/transcribe', methods = ['POST'])

def transcription():
    if "file" not in request.files:
        return(
            "Nothing received", 400
        )
    
    file = request.files["file"]

    with tempfile.NamedTemporaryFile(delete=False, suffix = ".wav") as temp:
        file.save(temp.name)
        audio_filename = temp.name

    try:

        result = model.transcribe(audio_filename)
        transcription = result["text"]

        if os.path.exists(audio_filename):
            os.unlink(audio_filename)
        
        return jsonify({
            "Result" : "Success",
            "Text"   : transcription
        })

    except Exception as e:
        if os.path.exists(audio_filename):
            os.unlink(audio_filename)

        return jsonify ({
            "Result" : "Faillure",
            "Text"   : str(e)
        })

if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5001)