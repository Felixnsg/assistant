# browser_tts.py - Browser-based TTS player with file-based logging
import http.server
import socketserver
import threading
import webbrowser
import json
import time
import os
import urllib.parse
import logging
from datetime import datetime
from interfaces import speech


# Add near the top with other imports
TEMP_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

# In the /upload-audio route:
temp_file = os.path.join(TEMP_AUDIO_DIR, "temp_audio.wav")

# Configure logging to file, instead of having the logs appear in the terminal I have them be in file.
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "tts_player.log")
logger = logging.getLogger("tts_player")
logger.setLevel(logging.INFO)

# File handler for logging
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Disable propagation to root logger (to avoid console output)
logger.propagate = False

# Configuration
TTS_SERVER = os.getenv("TTS_SERVER", "http://localhost:7851")
DEFAULT_VOICE = "female_01.wav"  # Initial default voice
DEFAULT_LANGUAGE = "en"
WEB_PORT = 8765
QUEUE_FILE = "tts_queue.json"
SETTINGS_FILE = "tts_settings.json"

# Available voices - you can add more based on what's available in your TTS system
AVAILABLE_VOICES = [
    {"id": "female_01.wav", "name": "Female 1"},
    {"id": "female_02.wav", "name": "Female 2"},
    {"id": "female_03.wav", "name": "Female 3"},
    {"id": "female_04.wav", "name": "Female 4"},
    {"id": "female_05.wav", "name": "Female 5"},
    {"id": "male_01.wav", "name": "Male 1"},
    {"id": "male_02.wav", "name": "Male 2"},
    {"id": "male_03.wav", "name": "Male 3"},
    {"id": "Morgan_Freeman CC3.wav", "name": "Morgan Freeman"}
]

# Available languages
AVAILABLE_LANGUAGES = [
    {"id": "en", "name": "English"},
    {"id": "fr", "name": "French"},
    {"id": "es", "name": "Spanish"},
    {"id": "de", "name": "German"},
    {"id": "it", "name": "Italian"},
    {"id": "pt", "name": "Portuguese"},
    {"id": "nl", "name": "Dutch"},
    {"id": "ru", "name": "Russian"},
    {"id": "ja", "name": "Japanese"},
    {"id": "zh", "name": "Chinese"}
]

# Audio state tracking event - NEW!
audio_finished_event = threading.Event()
# Initially set to "done" state
audio_finished_event.set()

# Singleton TTS Player Class
class TTSPlayer:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(TTSPlayer, cls).__new__(cls)
            cls._instance.server_started = False
            cls._instance.browser_opened = False
            cls._instance.server_thread = None
            cls._instance.current_voice = DEFAULT_VOICE
            cls._instance.current_language = DEFAULT_LANGUAGE
            cls._instance.is_playing = False  # NEW! track playing status
            cls._instance.ensure_files()
            cls._instance.load_settings()
        return cls._instance
    
    def ensure_files(self):
        """Make sure the queue and settings files exist"""
        if not os.path.exists(QUEUE_FILE):
            with open(QUEUE_FILE, 'w') as f:
                json.dump({"queue": []}, f)
            logger.info(f"Created queue file: {QUEUE_FILE}")
        
        if not os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'w') as f:
                json.dump({
                    "default_voice": DEFAULT_VOICE,
                    "default_language": DEFAULT_LANGUAGE
                }, f)
            logger.info(f"Created settings file: {SETTINGS_FILE}")
    
    def load_settings(self):
        """Load settings from the settings file"""
        self.ensure_files()
        
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                self.current_voice = settings.get("default_voice", DEFAULT_VOICE)
                self.current_language = settings.get("default_language", DEFAULT_LANGUAGE)
            logger.info(f"Settings loaded. Current voice: {self.current_voice}, Current language: {self.current_language}")
        except Exception as e:
            logger.error(f"Error loading settings: {e}")
    
    def save_settings(self):
        """Save current settings to the settings file"""
        self.ensure_files()
        
        try:
            with open(SETTINGS_FILE, 'w') as f:
                json.dump({
                    "default_voice": self.current_voice,
                    "default_language": self.current_language
                }, f)
            logger.info(f"Settings saved. Default voice: {self.current_voice}, Default language: {self.current_language}")
        except Exception as e:
            logger.error(f"Error saving settings: {e}")
    
    def get_available_voices(self):
        """Return a list of available voices"""
        return AVAILABLE_VOICES
    
    def get_available_languages(self):
        """Return a list of available languages"""
        return AVAILABLE_LANGUAGES
    
    def set_voice(self, voice_id):
        """
        Set the default voice to use for all TTS
        
        Args:
            voice_id (str): ID of the voice to use (e.g. "female_01.wav")
        
        Returns:
            bool: True if successful, False if voice not found
        """
        # Check if the voice exists
        voice_exists = any(voice["id"] == voice_id for voice in AVAILABLE_VOICES)
        
        if voice_exists:
            self.current_voice = voice_id
            self.save_settings()
            logger.info(f"Default voice set to: {voice_id}")
            return True
        else:
            logger.warning(f"Voice '{voice_id}' not found. Using current voice: {self.current_voice}")
            return False
    
    def set_language(self, language_id):
        """
        Set the default language to use for all TTS
        
        Args:
            language_id (str): ID of the language to use (e.g. "en")
        
        Returns:
            bool: True if successful, False if language not found
        """
        # Check if the language exists
        language_exists = any(language["id"] == language_id for language in AVAILABLE_LANGUAGES)
        
        if language_exists:
            self.current_language = language_id
            self.save_settings()
            logger.info(f"Default language set to: {language_id}")
            return True
        else:
            logger.warning(f"Language '{language_id}' not found. Using current language: {self.current_language}")
            return False
    
    def start_server(self):
        """Start the web server if it's not already running"""
        if not self.server_started:
            logger.info("Starting TTS player server...")
            
            player_instance = self  # Reference to the singleton for the handler class
            
            # Create a handler for the web server
            class TTSHandler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self):
                    # Serve main player page
                    if self.path == '/':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(get_player_html().encode())
                    elif self.path == '/record':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(get_recorder_html().encode())
                    # API to get the queue
                    elif self.path == '/queue':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        with open(QUEUE_FILE, 'r') as f:
                            self.wfile.write(f.read().encode())
                    
                    # API to get available voices
                    elif self.path == '/voices':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(AVAILABLE_VOICES).encode())
                    
                    # API to get current voice
                    elif self.path == '/current-voice':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"voice": player_instance.current_voice}).encode())
                    
                    # API to get available languages
                    elif self.path == '/languages':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps(AVAILABLE_LANGUAGES).encode())
                    
                    # API to get current language
                    elif self.path == '/current-language':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"language": player_instance.current_language}).encode())
                    
                    # NEW! API to get current audio status
                    elif self.path == '/audio-status':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "is_playing": player_instance.is_playing,
                            "has_queue_items": player_instance.has_items_in_queue()
                        }).encode())
                        # Add this in the do_GET method, with the other endpoints
                    elif self.path == '/record':
                        self.send_response(200)
                        self.send_header('Content-type', 'text/html')
                        self.end_headers()
                        self.wfile.write(get_recorder_html().encode())

                    # Add this in the do_POST method, with the other endpoints
                    elif self.path == '/upload-audio':
                        content_length = int(self.headers['Content-Length'])
                        audio_data = self.rfile.read(content_length)
                        
                        # Save to temporary file
                        temp_file = os.path.join(os.path.dirname(__file__), "temp_audio.wav")
                        with open(temp_file, 'wb') as f:
                            f.write(audio_data)
                        
                        logger.info(f"Received audio file, saved to {temp_file}")
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"success": True, "file": temp_file}).encode())
                                        
                    # API to get the next item
                    elif self.path == '/next':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        with open(QUEUE_FILE, 'r') as f:
                            queue_data = json.load(f)
                        
                        if queue_data["queue"]:
                            next_item = queue_data["queue"][0]
                            queue_data["queue"] = queue_data["queue"][1:]
                            
                            with open(QUEUE_FILE, 'w') as f:
                                json.dump(queue_data, f)
                            
                            self.wfile.write(json.dumps(next_item).encode())
                        else:
                            self.wfile.write(json.dumps({"empty": True}).encode())
                    
                    # All other paths
                    else:
                        return http.server.SimpleHTTPRequestHandler.do_GET(self)
                
                def do_POST(self):
                    # Get content length
                    content_length = int(self.headers['Content-Length']) if 'Content-Length' in self.headers else 0
                    
                    # API to set current voice
                    if self.path == '/set-voice':
                        post_data = self.rfile.read(content_length)
                        voice_data = json.loads(post_data)
                        
                        new_voice = voice_data.get("voice")
                        
                        if new_voice:
                            player_instance.set_voice(new_voice)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"success": True, "voice": player_instance.current_voice}).encode())
                    
                    # API to set current language
                    # Add this inside do_POST, alongside other routes
                    elif self.path == '/upload-audio':
                        content_length = int(self.headers['Content-Length'])
                        audio_data = self.rfile.read(content_length)
                        
                        # Create temp directory
                        temp_dir = os.path.join(os.path.dirname(__file__), "temp_audio")
                        os.makedirs(temp_dir, exist_ok=True)
                        
                        # Save to file
                        temp_file = os.path.join(temp_dir, "temp_audio.wav")
                        with open(temp_file, 'wb') as f:
                            f.write(audio_data)
                        
                        logger.info(f"Received audio file, saved to {temp_file}")
                        print(f"Received audio file, saved to {temp_file}")
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"success": True, "file": temp_file}).encode())
                    elif self.path == '/set-language':
                        post_data = self.rfile.read(content_length)
                        language_data = json.loads(post_data)
                        
                        new_language = language_data.get("language")
                        
                        if new_language:
                            player_instance.set_language(new_language)
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"success": True, "language": player_instance.current_language}).encode())
                    
                    # NEW! API to update audio playing status
                    elif self.path == '/set-audio-status':
                        post_data = self.rfile.read(content_length)
                        status_data = json.loads(post_data)
                        
                        is_playing = status_data.get("is_playing", False)
                        player_instance.is_playing = is_playing
                        
                        # Update the global event flag
                        if is_playing:
                            audio_finished_event.clear()  # Audio is playing
                            logger.info("Audio started playing")
                        else:
                            audio_finished_event.set()  # Audio is finished
                            logger.info("Audio finished playing")
                        
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({"success": True}).encode())
                    
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def do_DELETE(self):
                    # API to clear the queue
                    if self.path == '/queue':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        
                        with open(QUEUE_FILE, 'w') as f:
                            json.dump({"queue": []}, f)
                        
                        self.wfile.write(json.dumps({"success": True}).encode())
                    else:
                        self.send_response(404)
                        self.end_headers()
                
                def log_message(self, format, *args):
                    # Use our file logger instead of console
                    if not any(path in args[1] for path in ['/next', '/queue', '/voices', '/languages', '/current-voice', '/current-language', '/audio-status', '/set-audio-status']):
                        logger.info(f"HTTP: {format % args}")
            
            # Start the server
            try:
                socketserver.TCPServer.allow_reuse_address = True
                httpd = socketserver.TCPServer(("", WEB_PORT), TTSHandler)
                
                self.server_thread = threading.Thread(target=httpd.serve_forever)
                self.server_thread.daemon = True
                self.server_thread.start()
                self.server_started = True
                
                logger.info(f"TTS player server started on port {WEB_PORT}")
                logger.info(f"Using voice: {self.current_voice}, language: {self.current_language}")
            except Exception as e:
                logger.error(f"Error starting server: {e}")
    
    def say(self, text, voice=None, language=None):
        """
        Add text to the TTS queue to be spoken
        
        Args:
            text (str): Text to speak
            voice (str): Voice to use (defaults to current global voice)
            language (str): Language code (defaults to DEFAULT_LANGUAGE)
        """
        # Start server if needed
        if not self.server_started:
            self.start_server()
            time.sleep(0.5)  # Give the server a moment to start
        
        # Open browser if it hasn't been opened yet
        if not self.browser_opened:
            webbrowser.open(f"http://localhost:{WEB_PORT}")
            self.browser_opened = True
            logger.info(f"TTS player opened in browser at http://localhost:{WEB_PORT}")
        
        # Set defaults
        voice = voice or self.current_voice
        language = language or self.current_language
        
        # Update the global event flag - NEW!
        audio_finished_event.clear()  # Mark that audio will be playing
        
        # Add to queue
        with open(QUEUE_FILE, 'r') as f:
            queue_data = json.load(f)
        
        queue_data["queue"].append({
            "text": text,
            "voice": voice,
            "language": language,
            "timestamp": time.time()
        })
        
        with open(QUEUE_FILE, 'w') as f:
            json.dump(queue_data, f)
        
        logger.info(f"Added to TTS queue: {text[:50]}{'...' if len(text) > 50 else ''}")
        return True
    
    def clear_queue(self):
        """Clear the TTS queue"""
        self.ensure_files()
        with open(QUEUE_FILE, 'w') as f:
            json.dump({"queue": []}, f)
        logger.info("TTS queue cleared")
    
    def has_items_in_queue(self):
        """
        Check if there are items in the TTS queue
        
        Returns:
            bool: True if queue has items, False otherwise
        """
        self.ensure_files()
        try:
            with open(QUEUE_FILE, 'r') as f:
                queue_data = json.load(f)
            
            # Check if there are items in the queue
            return len(queue_data.get("queue", [])) > 0
        except Exception as e:
            logger.error(f"Error checking queue: {e}")
            return False  # Assume empty if there's an error

# NEW! Function to check if it's safe to listen
def is_safe_to_listen():
    """
    Check if it's safe for the microphone to listen
    
    Returns:
        bool: True if it's safe to listen (no audio playing), False otherwise
    """
    # Use the global event flag to check if audio is finished
    return audio_finished_event.is_set()

# NEW! Function to wait until it's safe to listen
def wait_until_safe_to_listen(timeout=30):
    """
    Wait until it's safe for the microphone to listen
    
    Args:
        timeout (int): Maximum time to wait in seconds
        
    Returns:
        bool: True if it's now safe to listen, False if timeout occurred
    """
    logger.info("Waiting for audio to finish before listening...")
    return audio_finished_event.wait(timeout=timeout)

# Wrapper functions to use the singleton class
def start_server():
    """Start the web server if it's not already running"""
    player = TTSPlayer()
    player.start_server()

def say(text, voice=None, language=None):
    """
    Add text to the TTS queue to be spoken
    
    Args:
        text (str): Text to speak
        voice (str): Voice to use (defaults to current voice)
        language (str): Language code (defaults to DEFAULT_LANGUAGE)
    """
    text = speech.clean_text_for_tts(text= text)
    text = str(text)
    
    player = TTSPlayer()
    
    # If this is the first time we're opening the browser,
    # add an extra delay to make sure it has time to open
    is_first_call = not player.browser_opened
    
    result = player.say(text, voice, language)
    
    if is_first_call:
        # Add an extra delay after the first call to ensure browser has time to open
        time.sleep(10)
    
    return result

def set_voice(voice_id):
    """
    Set the default voice to use for all TTS
    
    Args:
        voice_id (str): ID of the voice to use (e.g. "female_01.wav")
    
    Returns:
        bool: True if successful, False if voice not found
    """
    player = TTSPlayer()
    return player.set_voice(voice_id)

def set_language(language_id):
    """
    Set the default language to use for all TTS
    
    Args:
        language_id (str): ID of the language to use (e.g. "en")
    
    Returns:
        bool: True if successful, False if language not found
    """
    player = TTSPlayer()
    return player.set_language(language_id)

def get_available_voices():
    """Return a list of available voices"""
    player = TTSPlayer()
    return player.get_available_voices()

def get_available_languages():
    """Return a list of available languages"""
    player = TTSPlayer()
    return player.get_available_languages()

def clear_queue():
    """Clear the TTS queue"""
    player = TTSPlayer()
    player.clear_queue()

def has_items_in_queue():
    """
    Check if there are items in the TTS queue
    
    Returns:
        bool: True if queue has items, False otherwise
    """
    player = TTSPlayer()
    return player.has_items_in_queue()

# NEW! Helper functions
def is_audio_playing():
    """Check if audio is currently playing"""
    player = TTSPlayer()
    return player.is_playing

def get_player_html():
    """Return the HTML for the browser player with minimalist UI"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Player</title>
        <link rel="preconnect" href="https://fonts.googleapis.com">
        <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
        <link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@300;400&display=swap" rel="stylesheet">
        <style>
            :root {{
                /* Dark theme (default) */
                --bg-color: #121212;
                --text-color: rgba(255, 255, 255, 0.87);
                --glow-color: rgba(255, 255, 255, 0.9);
                --shadow-color: rgba(255, 255, 255, 0.4);
                --button-bg: rgba(255, 255, 255, 0.1);
                --button-hover: rgba(255, 255, 255, 0.2);
                --error-bg: rgba(220, 53, 69, 0.6);
            }}
            
            /* Light version */
            [data-theme="light"] {{
                --bg-color: #f5f5f5;
                --text-color: rgba(0, 0, 0, 0.87);
                --glow-color: rgba(0, 0, 0, 0.9);
                --shadow-color: rgba(0, 0, 0, 0.4);
                --button-bg: rgba(0, 0, 0, 0.08);
                --button-hover: rgba(0, 0, 0, 0.15);
                --error-bg: rgba(220, 53, 69, 0.6);
            }}
            
            /* Blue theme - dark version */
            [data-theme="blue"] {{
                --bg-color: #1a2a3a;
                --text-color: rgba(220, 240, 255, 0.9);
                --glow-color: rgba(100, 200, 255, 0.9);
                --shadow-color: rgba(100, 200, 255, 0.5);
                --button-bg: rgba(100, 200, 255, 0.15);
                --button-hover: rgba(100, 200, 255, 0.25);
                --error-bg: rgba(220, 53, 69, 0.6);
            }}
            
            /* Blue theme - light version */
            [data-theme="blue-light"] {{
                --bg-color: #e8f0f8;
                --text-color: rgba(25, 60, 110, 0.9);
                --glow-color: rgba(25, 120, 200, 0.9);
                --shadow-color: rgba(25, 120, 200, 0.4);
                --button-bg: rgba(25, 120, 200, 0.1);
                --button-hover: rgba(25, 120, 200, 0.2);
                --error-bg: rgba(220, 53, 69, 0.6);
            }}
            
            /* Beige theme - dark version */
            [data-theme="beige"] {{
                --bg-color: #2a2520;
                --text-color: rgba(255, 245, 230, 0.9);
                --glow-color: rgba(255, 230, 180, 0.9);
                --shadow-color: rgba(255, 230, 180, 0.5);
                --button-bg: rgba(255, 230, 180, 0.15);
                --button-hover: rgba(255, 230, 180, 0.25);
                --error-bg: rgba(220, 53, 69, 0.6);
            }}
            
            /* Beige theme - light version */
            [data-theme="beige-light"] {{
                --bg-color: #f8f4e8;
                --text-color: rgba(100, 70, 30, 0.9);
                --glow-color: rgba(190, 150, 80, 0.9);
                --shadow-color: rgba(190, 150, 80, 0.4);
                --button-bg: rgba(190, 150, 80, 0.15);
                --button-hover: rgba(190, 150, 80, 0.25);
                --error-bg: rgba(220, 53, 69, 0.6);
            }}
            
            /* Purple theme - dark version */
            [data-theme="purple"] {{
                --bg-color: #2a2035;
                --text-color: rgba(240, 230, 255, 0.9);
                --glow-color: rgba(180, 150, 240, 0.9);
                --shadow-color: rgba(180, 150, 240, 0.5);
                --button-bg: rgba(180, 150, 240, 0.15);
                --button-hover: rgba(180, 150, 240, 0.25);
                --error-bg: rgba(220, 53, 69, 0.6);
            }}
            
            /* Purple theme - light version */
            [data-theme="purple-light"] {{
                --bg-color: #f0e8f8;
                --text-color: rgba(90, 60, 140, 0.9);
                --glow-color: rgba(130, 90, 210, 0.9);
                --shadow-color: rgba(130, 90, 210, 0.4);
                --button-bg: rgba(130, 90, 210, 0.1);
                --button-hover: rgba(130, 90, 210, 0.2);
                --error-bg: rgba(220, 53, 69, 0.6);
            }}
            
            * {{ 
                margin: 0; 
                padding: 0; 
                box-sizing: border-box; 
            }}
            
            body {{ 
                font-family: 'Montserrat', sans-serif;
                background-color: var(--bg-color);
                color: var(--text-color);
                height: 100vh;
                width: 100vw;
                display: flex;
                justify-content: center;
                align-items: center;
                overflow: hidden;
                transition: all 0.3s ease;
            }}
            
            .circle-container {{
                position: relative;
                width: 400px;
                height: 400px;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            
            .circle {{
                width: 280px;
                height: 280px;
                border-radius: 50%;
                position: relative;
                cursor: pointer;
                background-color: transparent;
            }}
            
            .circle::before {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                border-radius: 50%;
                box-shadow: 0 0 40px 5px var(--shadow-color);
                opacity: 0.6;
            }}
            
            .circle::after {{
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                border-radius: 50%;
                border: 2px solid var(--glow-color);
                box-shadow: 0 0 20px var(--glow-color), inset 0 0 20px var(--glow-color);
            }}
            
            .playing .circle::after {{
                animation: pulse 1s infinite cubic-bezier(0.4, 0, 0.6, 1);
            }}
            
            .playing .circle::before {{
                animation: innerPulse 1.2s infinite ease-out alternate;
            }}
            
            @keyframes pulse {{
                0% {{ 
                    transform: scale(1); 
                    opacity: 1; 
                    box-shadow: 0 0 20px var(--glow-color), inset 0 0 20px var(--glow-color); 
                }}
                50% {{ 
                    transform: scale(1.1); 
                    opacity: 0.7; 
                    box-shadow: 0 0 35px var(--glow-color), inset 0 0 30px var(--glow-color); 
                }}
                100% {{ 
                    transform: scale(1); 
                    opacity: 1; 
                    box-shadow: 0 0 20px var(--glow-color), inset 0 0 20px var(--glow-color); 
                }}
            }}
            
            @keyframes innerPulse {{
                0% {{ 
                    opacity: 0.4; 
                    box-shadow: 0 0 25px var(--shadow-color);
                }}
                100% {{ 
                    opacity: 0.9; 
                    box-shadow: 0 0 45px var(--shadow-color);
                }}
            }}
            
            .corner-button {{
                position: absolute;
                width: 40px;
                height: 40px;
                border-radius: 50%;
                background-color: var(--button-bg);
                border: none;
                cursor: pointer;
                color: var(--text-color);
                font-size: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.3s ease;
            }}
            
            .corner-button:hover {{
                background-color: var(--button-hover);
            }}
            
            #clearBtn {{
                top: 20px;
                right: 20px;
            }}
            
            #themeToggle {{
                top: 20px;
                left: 20px;
            }}
            
            #queueInfo {{
                position: absolute;
                bottom: 20px;
                font-size: 14px;
                font-weight: 300;
                letter-spacing: 1px;
                opacity: 0.8;
            }}
            
            #viewQueue {{
                position: absolute;
                bottom: 20px;
                right: 20px;
                padding: 8px 16px;
                background-color: var(--button-bg);
                border: none;
                border-radius: 20px;
                color: var(--text-color);
                font-size: 12px;
                cursor: pointer;
                transition: all 0.3s ease;
            }}
            
            #viewQueue:hover {{
                background-color: var(--button-hover);
            }}
            
            .queue-details {{
                position: absolute;
                right: 20px;
                bottom: 60px;
                width: 280px;
                max-height: 200px;
                overflow-y: auto;
                background-color: var(--bg-color);
                box-shadow: 0 0 20px rgba(0, 0, 0, 0.2);
                border-radius: 8px;
                padding: 10px;
                z-index: 10;
                font-size: 12px;
                display: none;
            }}
            
            .queue-item {{
                padding: 8px;
                border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            }}
            
            .error-message {{
                position: absolute;
                bottom: -40px;
                background-color: var(--error-bg);
                color: var(--text-color);
                padding: 6px 12px;
                border-radius: 20px;
                max-width: 240px;
                font-size: 12px;
                text-align: center;
                opacity: 0;
                transition: opacity 0.5s ease;
            }}
            
            /* For the audio player - make it invisible */
            #audioPlayer {{
                opacity: 0;
                height: 0;
                width: 0;
                position: absolute;
            }}
        </style>
    </head>
    <body>
        <div class="circle-container">
            <div class="circle" id="mainCircle"></div>
            <div id="queueInfo">Ready</div>
            <div id="errorMessage" class="error-message"></div>
        </div>
        
        <button id="clearBtn" class="corner-button">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="3 6 5 6 21 6"></polyline>
                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
            </svg>
        </button>
        
        <button id="themeToggle" class="corner-button" title="Toggle Dark/Light">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="5"></circle>
                <line x1="12" y1="1" x2="12" y2="3"></line>
                <line x1="12" y1="21" x2="12" y2="23"></line>
                <line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line>
                <line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line>
                <line x1="1" y1="12" x2="3" y2="12"></line>
                <line x1="21" y1="12" x2="23" y2="12"></line>
                <line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line>
                <line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line>
            </svg>
        </button>
        
        <button id="colorThemeBtn" class="corner-button" style="top: 80px; left: 20px;" title="Change Color Theme">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <circle cx="12" cy="12" r="10"></circle>
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
            </svg>
        </button>
        
        <button id="voiceToggle" class="corner-button" style="top: 20px; left: 80px;" title="Change Voice">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"></path>
                <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
                <line x1="12" y1="19" x2="12" y2="23"></line>
                <line x1="8" y1="23" x2="16" y2="23"></line>
            </svg>
        </button>
        
        <button id="languageToggle" class="corner-button" style="top: 20px; left: 140px;" title="Change Language">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
                <path d="M2 12h20"></path>
            </svg>
        </button>
        
        <div id="voiceSelector" style="position: absolute; top: 70px; left: 80px; background-color: var(--bg-color); border-radius: 8px; padding: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); display: none; z-index: 10; min-width: 180px;">
            <select id="voiceSelect" style="width: 100%; padding: 8px; background-color: var(--button-bg); color: var(--text-color); border: none; border-radius: 4px; margin-bottom: 8px;"></select>
            <button id="setVoiceBtn" style="width: 100%; padding: 8px; background-color: var(--button-bg); color: var(--text-color); border: none; border-radius: 4px; cursor: pointer;">Apply Voice</button>
        </div>
        
        <div id="languageSelector" style="position: absolute; top: 70px; left: 140px; background-color: var(--bg-color); border-radius: 8px; padding: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.2); display: none; z-index: 10; min-width: 180px;">
            <select id="languageSelect" style="width: 100%; padding: 8px; background-color: var(--button-bg); color: var(--text-color); border: none; border-radius: 4px; margin-bottom: 8px;"></select>
            <button id="setLanguageBtn" style="width: 100%; padding: 8px; background-color: var(--button-bg); color: var(--text-color); border: none; border-radius: 4px; cursor: pointer;">Apply Language</button>
        </div>
        
        <button id="viewQueue">View Queue</button>
        <div id="queueDetails" class="queue-details"></div>
        
        <audio id="audioPlayer" controls>
            Your browser does not support the audio element.
        </audio>
        
        <script>
            // DOM elements
            const body = document.body;
            const audioPlayer = document.getElementById('audioPlayer');
            const mainCircle = document.getElementById('mainCircle');
            const clearBtn = document.getElementById('clearBtn');
            const themeToggle = document.getElementById('themeToggle');
            const colorThemeBtn = document.getElementById('colorThemeBtn');
            const queueInfo = document.getElementById('queueInfo');
            const viewQueue = document.getElementById('viewQueue');
            const queueDetails = document.getElementById('queueDetails');
            const errorMessage = document.getElementById('errorMessage');
            const circleContainer = document.querySelector('.circle-container');
            const voiceToggle = document.getElementById('voiceToggle');
            const voiceSelector = document.getElementById('voiceSelector');
            const voiceSelect = document.getElementById('voiceSelect');
            const setVoiceBtn = document.getElementById('setVoiceBtn');
            const languageToggle = document.getElementById('languageToggle');
            const languageSelector = document.getElementById('languageSelector');
            const languageSelect = document.getElementById('languageSelect');
            const setLanguageBtn = document.getElementById('setLanguageBtn');
            
            // State
            let isPlaying = false;
            let currentItem = null;
            let checking = false;
            let currentVoice = "";
            let currentLanguage = "en";
            let availableVoices = [];
            let availableLanguages = [];
            let isDarkMode = true;
            let currentColorTheme = "default"; // default, blue, beige, purple
            
            // Function to update server about audio playing status
            function updateAudioStatus(playing) {{
                fetch('/set-audio-status', {{
                    method: 'POST',
                    headers: {{
                        'Content-Type': 'application/json'
                    }},
                    body: JSON.stringify({{ is_playing: playing }})
                }})
                .then(response => response.json())
                .then(data => {{
                    console.log("Updated audio status:", playing);
                }})
                .catch(error => {{
                    console.error("Error updating audio status:", error);
                }});
            }}
            
            // Show error message for 15 seconds
            function showErrorMessage(message) {{
                errorMessage.textContent = message;
                errorMessage.style.opacity = 1;
                
                setTimeout(() => {{
                    errorMessage.style.opacity = 0;
                }}, 15000);
            }}
            
            // Theme management
            function updateTheme() {{
                if (isDarkMode) {{
                    if (currentColorTheme === "default") {{
                        body.removeAttribute('data-theme');
                    }} else {{
                        body.setAttribute('data-theme', currentColorTheme);
                    }}
                }} else {{
                    // Light versions
                    if (currentColorTheme === "default") {{
                        body.setAttribute('data-theme', 'light');
                    }} else {{
                        body.setAttribute('data-theme', `${{currentColorTheme}}-light`);
                    }}
                }}
                
                // Update selector themes
                updateVoiceSelectorTheme();
                updateLanguageSelectorTheme();
            }}
            
            function updateVoiceSelectorTheme() {{
                if (voiceSelector.style.display === 'block') {{
                    voiceSelector.style.backgroundColor = getComputedStyle(body).getPropertyValue('--bg-color');
                    voiceSelect.style.backgroundColor = getComputedStyle(body).getPropertyValue('--button-bg');
                    voiceSelect.style.color = getComputedStyle(body).getPropertyValue('--text-color');
                    setVoiceBtn.style.backgroundColor = getComputedStyle(body).getPropertyValue('--button-bg');
                    setVoiceBtn.style.color = getComputedStyle(body).getPropertyValue('--text-color');
                }}
            }}
            
            function updateLanguageSelectorTheme() {{
                if (languageSelector.style.display === 'block') {{
                    languageSelector.style.backgroundColor = getComputedStyle(body).getPropertyValue('--bg-color');
                    languageSelect.style.backgroundColor = getComputedStyle(body).getPropertyValue('--button-bg');
                    languageSelect.style.color = getComputedStyle(body).getPropertyValue('--text-color');
                    setLanguageBtn.style.backgroundColor = getComputedStyle(body).getPropertyValue('--button-bg');
                    setLanguageBtn.style.color = getComputedStyle(body).getPropertyValue('--text-color');
                }}
            }}
            
            // Dark/Light toggle
            themeToggle.addEventListener('click', () => {{
                isDarkMode = !isDarkMode;
                updateTheme();
            }});
            
            // Color theme toggle
            colorThemeBtn.addEventListener('click', () => {{
                // Cycle through color themes: default -> blue -> beige -> purple -> default
                switch(currentColorTheme) {{
                    case "default":
                        currentColorTheme = "blue";
                        break;
                    case "blue":
                        currentColorTheme = "beige";
                        break;
                    case "beige":
                        currentColorTheme = "purple";
                        break;
                    case "purple":
                        currentColorTheme = "default";
                        break;
                }}
                updateTheme();
            }});
            
            // Voice selector toggle with debug
            voiceToggle.addEventListener('click', function(e) {{
                console.log("Voice toggle clicked");
                if (voiceSelector.style.display === 'block') {{
                    console.log("Hiding voice selector");
                    voiceSelector.style.display = 'none';
                }} else {{
                    console.log("Showing voice selector");
                    voiceSelector.style.display = 'block';
                    // Hide language selector if it's open
                    languageSelector.style.display = 'none';
                    // Force re-flow
                    void voiceSelector.offsetWidth;
                    updateVoiceSelectorTheme();
                }}
                e.stopPropagation(); // Prevent event bubbling
            }});
            
            // Language selector toggle
            languageToggle.addEventListener('click', function(e) {{
                console.log("Language toggle clicked");
                if (languageSelector.style.display === 'block') {{
                    console.log("Hiding language selector");
                    languageSelector.style.display = 'none';
                }} else {{
                    console.log("Showing language selector");
                    languageSelector.style.display = 'block';
                    // Hide voice selector if it's open
                    voiceSelector.style.display = 'none';
                    // Force re-flow
                    void languageSelector.offsetWidth;
                    updateLanguageSelectorTheme();
                }}
                e.stopPropagation(); // Prevent event bubbling
            }});
            
            // Close selectors when clicking elsewhere
            document.addEventListener('click', function(e) {{
                if (voiceSelector.style.display === 'block' && 
                    !voiceSelector.contains(e.target) && 
                    e.target !== voiceToggle) {{
                    voiceSelector.style.display = 'none';
                }}
                
                if (languageSelector.style.display === 'block' && 
                    !languageSelector.contains(e.target) && 
                    e.target !== languageToggle) {{
                    languageSelector.style.display = 'none';
                }}
            }});
            
            // View queue toggle
            viewQueue.addEventListener('click', () => {{
                if (queueDetails.style.display === 'block') {{
                    queueDetails.style.display = 'none';
                }} else {{
                    queueDetails.style.display = 'block';
                    updateQueueDisplay(); // Make sure it's updated
                }}
            }});
            
            // Load voices
            async function loadVoices() {{
                try {{
                    // Get available voices
                    const voicesResponse = await fetch('/voices');
                    availableVoices = await voicesResponse.json();
                    
                    // Get current voice
                    const currentVoiceResponse = await fetch('/current-voice');
                    const currentVoiceData = await currentVoiceResponse.json();
                    currentVoice = currentVoiceData.voice;
                    
                    // Populate voice selector
                    voiceSelect.innerHTML = '';
                    availableVoices.forEach(voice => {{
                        const option = document.createElement('option');
                        option.value = voice.id;
                        option.text = voice.name;
                        if (voice.id === currentVoice) {{
                            option.selected = true;
                        }}
                        voiceSelect.appendChild(option);
                    }});
                    
                    console.log("Voices loaded. Current voice:", currentVoice);
                }} catch (error) {{
                    console.error("Error loading voices:", error);
                    showErrorMessage("Failed to load voices");
                }}
            }}
            
            // Load languages
            async function loadLanguages() {{
                try {{
                    // Get available languages
                    const languagesResponse = await fetch('/languages');
                    availableLanguages = await languagesResponse.json();
                    
                    // Get current language
                    const currentLanguageResponse = await fetch('/current-language');
                    const currentLanguageData = await currentLanguageResponse.json();
                    currentLanguage = currentLanguageData.language;
                    
                    // Populate language selector
                    languageSelect.innerHTML = '';
                    availableLanguages.forEach(language => {{
                        const option = document.createElement('option');
                        option.value = language.id;
                        option.text = language.name;
                        if (language.id === currentLanguage) {{
                            option.selected = true;
                        }}
                        languageSelect.appendChild(option);
                    }});
                    
                    console.log("Languages loaded. Current language:", currentLanguage);
                }} catch (error) {{
                    console.error("Error loading languages:", error);
                    showErrorMessage("Failed to load languages");
                }}
            }}
            
            // Set voice
            async function setVoice() {{
                const newVoice = voiceSelect.value;
                
                try {{
                    const response = await fetch('/set-voice', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ voice: newVoice }})
                    }});
                    
                    const data = await response.json();
                    if (data.success) {{
                        currentVoice = data.voice;
                        showErrorMessage(`Voice set to: ${{newVoice.replace('.wav', '')}}`);
                        voiceSelector.style.display = 'none'; // Hide after setting
                        console.log("Voice set to:", currentVoice);
                    }}
                }} catch (error) {{
                    console.error("Error setting voice:", error);
                    showErrorMessage(`Error setting voice: ${{error.message}}`);
                }}
            }}
            
            // Set language
            async function setLanguage() {{
                const newLanguage = languageSelect.value;
                
                try {{
                    const response = await fetch('/set-language', {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json'
                        }},
                        body: JSON.stringify({{ language: newLanguage }})
                    }});
                    
                    const data = await response.json();
                    if (data.success) {{
                        currentLanguage = data.language;
                        showErrorMessage(`Language set to: ${{newLanguage}}`);
                        languageSelector.style.display = 'none'; // Hide after setting
                        console.log("Language set to:", currentLanguage);
                    }}
                }} catch (error) {{
                    console.error("Error setting language:", error);
                    showErrorMessage(`Error setting language: ${{error.message}}`);
                }}
            }}
            
            // Voice selector events with debug
            setVoiceBtn.addEventListener('click', function(e) {{
                console.log("Set voice button clicked");
                setVoice();
                e.stopPropagation(); // Prevent event bubbling
            }});
            
            // Language selector events
            setLanguageBtn.addEventListener('click', function(e) {{
                console.log("Set language button clicked");
                setLanguage();
                e.stopPropagation(); // Prevent event bubbling
            }});
            
            // Check for new items every 1 second
            const CHECK_INTERVAL = 1000;
            
            // Play audio
            function playAudio(item) {{
                currentItem = item;
                const text = item.text;
                const voice = item.voice;
                const language = item.language || 'en';
                
                console.log("Playing audio:", item);
                circleContainer.classList.add('playing');
                
                // Create TTS URL with GET parameters
                const params = new URLSearchParams();
                params.append('text', text);
                params.append('voice', voice);
                params.append('language', language);
                params.append('output_file', `tts_output_${{Date.now()}}.wav`);
                
                const url = `{TTS_SERVER}/api/tts-generate-streaming?${{params.toString()}}`;
                console.log("Generated URL:", url);
                
                // Update player
                try {{
                    audioPlayer.src = url;
                    
                    // Play and catch any errors
                    audioPlayer.play()
                        .then(() => {{
                            console.log("Play started");
                            isPlaying = true;
                            // Update server about playing status
                            updateAudioStatus(true);
                        }})
                        .catch(e => {{
                            console.error("Play error:", e);
                            showErrorMessage("Failed to play audio. Server might be unavailable.");
                            isPlaying = false;
                            updateAudioStatus(false);
                            circleContainer.classList.remove('playing');
                            currentItem = null;
                            // Try next item after error
                            setTimeout(checkQueue, 1000);
                        }});
                }} catch (e) {{
                    console.error("Error setting audio source:", e);
                    showErrorMessage("Error connecting to TTS server");
                    isPlaying = false;
                    updateAudioStatus(false);
                    circleContainer.classList.remove('playing');
                }}
                
                updateQueueDisplay();
            }}
            
            // Check for new items in the queue
            async function checkQueue() {{
                if (checking) return;
                checking = true;
                
                try {{
                    // Get next item if we're not currently playing
                    if (!isPlaying) {{
                        const response = await fetch('/next');
                        const data = await response.json();
                        
                        if (!data.empty) {{
                            playAudio(data);
                        }} else {{
                            queueInfo.textContent = "Queue empty";
                            circleContainer.classList.remove('playing');
                        }}
                    }}
                    
                    // Update queue display
                    updateQueueDisplay();
                }} catch (error) {{
                    console.error("Error checking queue:", error);
                    showErrorMessage("Error checking queue");
                }} finally {{
                    checking = false;
                }}
            }}
            
            // Update the queue display
            async function updateQueueDisplay() {{
                try {{
                    const response = await fetch('/queue');
                    const data = await response.json();
                    
                    // Update queue count
                    const queueCount = data.queue ? data.queue.length : 0;
                    
                    // Update queue info text
                    if (currentItem) {{
                        queueInfo.textContent = queueCount > 0 ? 
                            `${{queueCount}} text${{queueCount !== 1 ? 's' : ''}} left to play` : 
                            "Now playing";
                    }} else if (queueCount > 0) {{
                        queueInfo.textContent = `${{queueCount}} text${{queueCount !== 1 ? 's' : ''}} in queue`;
                    }} else {{
                        queueInfo.textContent = "Queue empty";
                    }}
                    
                    // Update detailed queue view if visible
                    if (queueDetails.style.display === 'block') {{
                        // Clear the queue details
                        queueDetails.innerHTML = '';
                        
                        // Add current item if playing
                        if (currentItem) {{
                            const currentItemDiv = document.createElement('div');
                            currentItemDiv.className = 'queue-item';
                            currentItemDiv.innerHTML = `<strong>Now playing:</strong> ${{currentItem.text.substring(0, 50)}}${{currentItem.text.length > 50 ? '...' : ''}}`;
                            queueDetails.appendChild(currentItemDiv);
                        }}
                        
                        // Add queue items
                        if (data.queue && data.queue.length > 0) {{
                            data.queue.forEach((item, index) => {{
                                const itemDiv = document.createElement('div');
                                itemDiv.className = 'queue-item';
                                itemDiv.innerHTML = `<strong>${{index + 1}}:</strong> ${{item.text.substring(0, 50)}}${{item.text.length > 50 ? '...' : ''}}`;
                                queueDetails.appendChild(itemDiv);
                            }});
                        }} else if (!currentItem) {{
                            queueDetails.innerHTML = '<div class="queue-item">No items in queue</div>';
                        }}
                    }}
                    
                }} catch (error) {{
                    console.error("Error updating queue display:", error);
                }}
            }}
            
            // Event listeners
            audioPlayer.addEventListener('error', (e) => {{
                console.error("Audio error event:", e);
                showErrorMessage("Error playing audio");
                isPlaying = false;
                updateAudioStatus(false);
                circleContainer.classList.remove('playing');
                currentItem = null;
                // Try next item after error
                setTimeout(checkQueue, 1000);
            }});
            
            audioPlayer.addEventListener('ended', () => {{
                console.log("Audio ended");
                isPlaying = false;
                // Update server about ended status
                updateAudioStatus(false);
                currentItem = null;
                circleContainer.classList.remove('playing');
                // Check for more items immediately
                checkQueue();
            }});
            
            // Toggle play/pause when clicking on the circle
            mainCircle.addEventListener('click', () => {{
                if (isPlaying) {{
                    audioPlayer.pause();
                    isPlaying = false;
                    updateAudioStatus(false);
                    circleContainer.classList.remove('playing');
                }} else {{
                    if (audioPlayer.src) {{
                        audioPlayer.play().catch(e => {{
                            console.error("Play error:", e);
                            showErrorMessage("Failed to play audio");
                        }});
                        isPlaying = true;
                        updateAudioStatus(true);
                        circleContainer.classList.add('playing');
                    }} else {{
                        checkQueue();
                    }}
                }}
            }});
            
            clearBtn.addEventListener('click', async () => {{
                try {{
                    await fetch('/queue', {{ method: 'DELETE' }});
                    updateQueueDisplay();
                    queueInfo.textContent = "Queue cleared";
                }} catch (error) {{
                    console.error("Error clearing queue:", error);
                    showErrorMessage("Error clearing queue");
                }}
            }});
            
            function restartPulseAnimation() {{
                // First remove the class
                circleContainer.classList.remove('playing');
                
                // Force a browser reflow
                void circleContainer.offsetWidth;
                
                // If we should be playing, add the class back
                if (isPlaying) {{
                    // Add with a tiny delay to ensure the class change registers
                    setTimeout(() => {{
                        circleContainer.classList.add('playing');
                    }}, 10);
                }}
            }}

            // In your playAudio function, replace this line:
            circleContainer.classList.add('playing');

            // With this:
            restartPulseAnimation();

            // Add this to your checkQueue function at the end:
            // Try to restart animation if we're playing but animation doesn't look right
            if (isPlaying && currentItem) {{
                restartPulseAnimation();
            }}

            // Add this interval to periodically restart animation if needed
            setInterval(() => {{
                if (isPlaying && currentItem) {{
                    // Check if animation seems stuck by examining if the transforms are changing
                    // This is a backup to ensure animation keeps running
                    restartPulseAnimation();
                    console.log("Restarting pulse animation as preventative measure");
                }}
            }}, 5000); // Every 5 seconds
            
            function isAudioActuallyPlaying() {{
                return !audioPlayer.paused && !audioPlayer.ended && audioPlayer.currentTime > 0;
            }}

            // Check if we need to restart animation when DOM is fully loaded
            document.addEventListener('DOMContentLoaded', function() {{
                console.log("DOM fully loaded");
                updateTheme();
                
                // Check if we should be playing
                if (isAudioActuallyPlaying()) {{
                    console.log("Audio is already playing on load, ensuring animation is active");
                    isPlaying = true;
                    restartPulseAnimation();
                }}
            }});

            // Check for animation state when window gains focus
            window.addEventListener('focus', function() {{
                console.log("Window gained focus, checking animation state");
                if (isAudioActuallyPlaying() && !circleContainer.classList.contains('playing')) {{
                    console.log("Audio is playing but animation is not, fixing...");
                    isPlaying = true;
                    restartPulseAnimation();
                }}
            }});

            // Add a click handler to the document to restart animation if needed
            document.addEventListener('click', function() {{
                if (isPlaying && !circleContainer.classList.contains('playing')) {{
                    console.log("Detected click while playing but animation is off, restarting...");
                    restartPulseAnimation();
                }}
            }});

            // Fix animation if browser was inactive
            document.addEventListener('visibilitychange', function() {{
                if (!document.hidden && isAudioActuallyPlaying()) {{
                    console.log("Page became visible, ensuring animation is running");
                    isPlaying = true;
                    restartPulseAnimation();
                }}
            }});

            // Extra check when audio time updates
            audioPlayer.addEventListener('timeupdate', function() {{
                // If audio is playing but animation is not showing, fix it
                if (isAudioActuallyPlaying() && !circleContainer.classList.contains('playing')) {{
                    console.log("Audio is playing but animation stopped, restarting...");
                    isPlaying = true;
                    restartPulseAnimation();
                }}
            }});
            
            // Load data on page load
            loadVoices();
            loadLanguages();
            
            // Initialize theme on load
            document.addEventListener('DOMContentLoaded', function() {{
                console.log("DOM fully loaded");
                updateTheme();
            }});
            
            // Start immediately
            setTimeout(checkQueue, 500);
            
            // Start checking the queue regularly
            setInterval(checkQueue, CHECK_INTERVAL);
                        
            // Add debug message
            console.log("Player initialized with:", {{
                themeToggle: !!themeToggle,
                colorThemeBtn: !!colorThemeBtn,
                voiceToggle: !!voiceToggle,
                languageToggle: !!languageToggle
            }});
        </script>
    </body>
    </html>
    """
# Add this function at the end of the file, near get_player_html()
def get_recorder_html():
    """Return the HTML for the browser recorder"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Voice Recorder</title>
        <style>
            body {
                font-family: 'Montserrat', sans-serif;
                background-color: #121212;
                color: rgba(255, 255, 255, 0.87);
                height: 100vh;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            .recorder-container {
                width: 300px;
                text-align: center;
            }
            
            .record-button {
                width: 80px;
                height: 80px;
                border-radius: 50%;
                background-color: #ff4b4b;
                margin: 20px auto;
                cursor: pointer;
                display: flex;
                justify-content: center;
                align-items: center;
                box-shadow: 0 0 10px rgba(255, 75, 75, 0.5);
                transition: all 0.3s ease;
            }
            
            .record-button.recording {
                animation: pulse 1.5s infinite;
                background-color: #ff0000;
            }
            
            @keyframes pulse {
                0% { transform: scale(1); }
                50% { transform: scale(1.1); }
                100% { transform: scale(1); }
            }
            
            .status {
                margin-top: 20px;
                font-size: 14px;
            }
            
            .timer {
                font-size: 24px;
                margin-top: 10px;
            }
        </style>
    </head>
    <body>
        <div class="recorder-container">
            <h2>Voice Recorder</h2>
            <div class="record-button" id="recordButton">
                <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2">
                    <circle cx="12" cy="12" r="6"></circle>
                </svg>
            </div>
            <div class="timer" id="timer">00:00</div>
            <div class="status" id="status">Click to start recording</div>
        </div>
        
        <script>
            const recordButton = document.getElementById('recordButton');
            const timer = document.getElementById('timer');
            const status = document.getElementById('status');
            
            let mediaRecorder;
            let audioChunks = [];
            let isRecording = false;
            let startTime;
            let timerInterval;
            
            // Request microphone access
            async function setupRecorder() {
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    mediaRecorder = new MediaRecorder(stream);
                    
                    mediaRecorder.ondataavailable = (event) => {
                        audioChunks.push(event.data);
                    };
                    
                    mediaRecorder.onstop = () => {
                        const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                        sendAudioToServer(audioBlob);
                    };
                    
                    status.textContent = "Ready to record";
                } catch (err) {
                    console.error("Error accessing microphone:", err);
                    status.textContent = "Error: Could not access microphone";
                }
            }
            
            // Send recorded audio to the server
            function sendAudioToServer(blob) {
                status.textContent = "Processing...";
                
                fetch('/upload-audio', {
                    method: 'POST',
                    body: blob
                })
                .then(response => response.json())
                .then(data => {
                    console.log("Server response:", data);
                    status.textContent = "Audio sent to server successfully";
                    setTimeout(() => {
                        status.textContent = "Click to start recording";
                    }, 3000);
                })
                .catch(error => {
                    console.error("Error sending audio:", error);
                    status.textContent = "Error sending audio to server";
                });
            }
            
            // Update timer display
            function updateTimer() {
                const elapsedTime = Math.floor((Date.now() - startTime) / 1000);
                const minutes = Math.floor(elapsedTime / 60).toString().padStart(2, '0');
                const seconds = (elapsedTime % 60).toString().padStart(2, '0');
                timer.textContent = `${minutes}:${seconds}`;
                
                // Auto-stop after 30 seconds
                if (elapsedTime >= 30 && isRecording) {
                    stopRecording();
                }
            }
            
            // Start recording
            function startRecording() {
                audioChunks = [];
                mediaRecorder.start();
                isRecording = true;
                recordButton.classList.add('recording');
                status.textContent = "Recording...";
                
                startTime = Date.now();
                timerInterval = setInterval(updateTimer, 1000);
            }
            
            // Stop recording
            function stopRecording() {
                if (isRecording) {
                    mediaRecorder.stop();
                    isRecording = false;
                    recordButton.classList.remove('recording');
                    clearInterval(timerInterval);
                }
            }
            
            // Toggle recording
            recordButton.addEventListener('click', () => {
                if (!mediaRecorder) {
                    status.textContent = "Waiting for microphone access...";
                    setupRecorder();
                    return;
                }
                
                if (isRecording) {
                    stopRecording();
                } else {
                    startRecording();
                }
            });
            
            // Initialize on page load
            window.addEventListener('load', setupRecorder);
        </script>
    </body>
    </html>
    """
# Create a simple function to show minimal info for the user
def status_message(message):
    """Show a status message to the user without detailed logging"""
    print(message)
    logger.info(message)

# Example usage
if __name__ == "__main__":
    status_message("TTS Queue System with Voice and Language Selection")
    status_message("------------------------------------")
    
    # List available voices
    status_message("\nAvailable voices:")
    for voice in AVAILABLE_VOICES:
        status_message(f"- {voice['name']} (ID: {voice['id']})")
    
    # List available languages
    status_message("\nAvailable languages:")
    for language in AVAILABLE_LANGUAGES:
        status_message(f"- {language['name']} (ID: {language['id']})")
    
    # Load settings
    player = TTSPlayer()
    status_message(f"\nCurrent voice: {player.current_voice}")
    status_message(f"Current language: {player.current_language}")
    
    # Option to change voice
    voice_choice = input("\nSelect a voice ID (or press Enter to keep current): ")
    if voice_choice:
        set_voice(voice_choice)
    
    # Option to change language
    language_choice = input("\nSelect a language ID (or press Enter to keep current): ")
    if language_choice:
        set_language(language_choice)
    
    # Start the server and open browser
    say("Hello, this is a test of the TTS queue system with voice and language selection.")
    time.sleep(1)
    say("The browser window will open. Click play to start the first message.")
    time.sleep(1)
    say("After the first message plays, the rest will play automatically.")
    say("You can change the voice and language in the browser interface.")
    
    status_message("\nBrowser has been opened. Click 'Play' button to start.")
    status_message("This window will stay open to process the TTS queue.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        status_message("Exiting...")