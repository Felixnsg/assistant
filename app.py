# app.py - Flask application
from flask import Flask, render_template, request, jsonify
import os
import sys

# Add path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Override the format choice to avoid terminal input prompt
# We need to do this BEFORE importing your modules
import builtins
original_input = builtins.input

# Override the input function to automatically return "text" when asked for format
def patched_input(prompt=""):
    if "format" in prompt.lower():
        return "text"  # Always return "text" for format questions
    return original_input(prompt)  # Use original input for other prompts
    
builtins.input = patched_input

# Now import your modules after we've patched the input function
from core import nlp
from core import memory
from interfaces import speech
from services import utilities
import config

app = Flask(__name__)

# Store the HTML UI in the templates folder
@app.route('/')
def index():
    return render_template('sypher-ui.html')

@app.route('/api/send-message', methods=['POST'])
def send_message():
    data = request.json
    message = data.get('message', '')
    
    # Use your existing text_convos logic, but modified for API use
    convos = memory.get_convos()
    past_messages = memory.get_convos()
    convos.extend(past_messages)
    
    convos.append({"role": "user", "parts": [{"text": message}]})
    data_for_ai = {
        "contents": convos,
        "generationConfig": {
            "temperature": 0.72
        }
    }
    
    # Check if weather information is needed
    needs_weather = False
    weather_info = None
    
    ai_response = nlp.send_request(data_for_ai)
    
    if utilities.monitor_sypher(ai_response):
        needs_weather = True
        weather_info = utilities.get_weather()
        
        convos.append({"role": "user", "parts": [{"text": config.weather_prompt + str(weather_info)}]})
        data_for_ai = {
            "contents": convos,
            "generationConfig": {
                "temperature": 0.72
            }
        }
        
        ai_response = nlp.send_request(data_for_ai)
    
    # Save conversation
    memory.save_convos("user", message)
    memory.save_convos("model", ai_response)
    
    # Return the response to the UI
    return jsonify({
        'response': ai_response,
        'needs_weather': needs_weather,
        'weather_info': weather_info
    })

@app.route('/api/voice-recognition', methods=['POST'])
def voice_recognition():
    # This would integrate with your speech recognition
    try:
        text = speech.speach_recognition()
        return jsonify({'text': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/text-to-speech', methods=['POST'])
def text_to_speech():
    data = request.json
    text = data.get('text', '')
    voice_type = data.get('voice', 'edge')
    
    try:
        # Call the appropriate TTS function based on the voice type
        if voice_type == 'edge':
            speech.edge_text_to_speech(text)
        elif voice_type == 'elevenlabs':
            speech.elevenLab_text_to_speech(text)
        elif voice_type == 'aws':
            speech.AWS_text_to_speech(text)
        else:
            speech.text_to_speech(text)  # Default
            
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear-memory', methods=['POST'])
def clear_memory():
    memory.delete_convos()
    return jsonify({'success': True})

@app.route('/api/get-history', methods=['GET'])
def get_history():
    convos = memory.get_convos()
    formatted_convos = []
    
    # Format the conversation history
    # This assumes your conversations are stored as alternating user/model messages
    for i in range(0, len(convos), 2):
        if i+1 < len(convos) and convos[i]['role'] == 'user':
            formatted_convos.append({
                'user': convos[i]['parts'][0]['text'] if 'parts' in convos[i] and len(convos[i]['parts']) > 0 else '',
                'response': convos[i+1]['parts'][0]['text'] if 'parts' in convos[i+1] and len(convos[i+1]['parts']) > 0 else '',
                'timestamp': 'Previous'  # You might want to add actual timestamps to your convos
            })
    
    return jsonify(formatted_convos)

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run the app
    app.run(debug=True, port=5000)


     








