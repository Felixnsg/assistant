# File: StreamTTSPlayer.py
"""
Provides TTS functionality using AllTalk server and ffplay for playback.
"""

import os
import sys
import time
import json
import logging
import tempfile
import threading
import subprocess
import requests
from typing import Optional, Dict, Any, List
from urllib.parse import quote

# Import required modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import config
except ImportError as e:
    print(f"Error importing dependencies: {e}")
    print("Ensure config.py is accessible.")
    sys.exit(1)

# Configure logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
logger = logging.getLogger("StreamTTSPlayer")
logger.setLevel(logging.INFO)
if not logger.handlers:
    file_handler = logging.FileHandler(os.path.join(log_dir, "stream_tts.log"), encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("StreamTTS: %(message)s"))
    logger.addHandler(console_handler)
logger.propagate = False

# Configuration
TTS_SERVER_URL = getattr(config, "TTS_SERVER", "http://localhost:7851/api/tts-generate-streaming")
SETTINGS_FILE = "tts_settings.json"
DEFAULT_VOICE = "female_01.wav"
DEFAULT_LANGUAGE = "en"

# Check for ffplay availability
def check_ffplay():
    """Check if ffplay is available on the system."""
    try:
        result = subprocess.run(
            ["ffplay", "-version"], 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False

# Audio playback availability
FFPLAY_AVAILABLE = check_ffplay()
if FFPLAY_AVAILABLE:
    logger.info("ffplay found and available for audio playback")
    print("Audio system initialized using ffplay")
else:
    logger.error("ffplay not found. Please install ffmpeg/ffplay.")
    print("Warning: Audio playback won't work. Please install ffmpeg/ffplay.")

# Simple text cleaning function
def clean_text_for_tts(text):
    """Basic text cleaning for TTS without external dependencies."""
    if not isinstance(text, str):
        return ""
    
    # Simple replacements for common problematic characters
    text = text.replace('**', '')
    text = text.replace('*', '')
    text = text.replace('_', '')
    text = text.replace('`', '')
    text = text.replace('#', '')
    
    return text.strip()

# Available voices and languages
AVAILABLE_VOICES = [
    {"id": "female_01.wav", "name": "Female 1"}, {"id": "female_02.wav", "name": "Female 2"},
    {"id": "female_03.wav", "name": "Female 3"}, {"id": "female_04.wav", "name": "Female 4"},
    {"id": "female_05.wav", "name": "Female 5"}, {"id": "male_01.wav", "name": "Male 1"},
    {"id": "male_02.wav", "name": "Male 2"}, {"id": "male_03.wav", "name": "Male 3"},
    {"id": "Morgan_Freeman CC3.wav", "name": "Morgan Freeman"}
]

AVAILABLE_LANGUAGES = [
    {"id": "en", "name": "English"}, {"id": "fr", "name": "French"}, 
    {"id": "es", "name": "Spanish"}, {"id": "de", "name": "German"}, 
    {"id": "it", "name": "Italian"}, {"id": "pt", "name": "Portuguese"},
    {"id": "nl", "name": "Dutch"}, {"id": "ru", "name": "Russian"}, 
    {"id": "ja", "name": "Japanese"}, {"id": "zh", "name": "Chinese"}
]

class StreamTTSPlayer:
    """
    Handles text-to-speech using AllTalk server and ffplay for playback.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating new StreamTTSPlayer instance")
            cls._instance = super(StreamTTSPlayer, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the player instance."""
        # Voice and language settings
        self.current_voice = DEFAULT_VOICE
        self.current_language = DEFAULT_LANGUAGE
        
        # Load saved settings
        self._load_settings()
        
        # Playback state
        self._is_playing = False
        self._playback_process = None
        self._playback_finished_event = threading.Event()
        self._playback_finished_event.set()  # Initially set (nothing playing)
        
        # Request queue and worker thread
        self._request_queue = []
        self._queue_lock = threading.Lock()
        self._worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self._worker_thread.start()
        
        logger.info(f"StreamTTSPlayer initialized (Voice: {self.current_voice}, Language: {self.current_language})")

    def _load_settings(self):
        """Load settings from file or create with defaults."""
        if not os.path.exists(SETTINGS_FILE):
            self._save_settings()  # Create default settings file
            return
            
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                settings = json.load(f)
                self.current_voice = settings.get("default_voice", DEFAULT_VOICE)
                self.current_language = settings.get("default_language", DEFAULT_LANGUAGE)
            logger.info(f"Settings loaded: Voice={self.current_voice}, Language={self.current_language}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading settings: {e}. Using defaults.")

    def _save_settings(self):
        """Save current settings to file."""
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump({
                    "default_voice": self.current_voice,
                    "default_language": self.current_language
                }, f, indent=2)
            logger.info(f"Settings saved: Voice={self.current_voice}, Language={self.current_language}")
        except IOError as e:
            logger.error(f"Error saving settings: {e}")

    def _process_queue(self):
        """Worker thread to process TTS requests from the queue."""
        logger.info("Queue processor thread started")
        
        while True:
            # Check if there's anything in the queue
            request = None
            with self._queue_lock:
                if self._request_queue:
                    request = self._request_queue.pop(0)
            
            # Process the request if we got one
            if request:
                text, voice, language = request
                self._process_tts_request(text, voice, language)
            else:
                # Sleep briefly if queue is empty
                time.sleep(0.1)

    def _process_tts_request(self, text, voice, language):
        """Process a single TTS request."""
        # Signal that playback is starting
        self._playback_finished_event.clear()
        self._is_playing = True
        
        try:
            # Create URL with query parameters
            encoded_text = quote(text)
            output_file = f"stream_{int(time.time())}.wav"
            url = f"{TTS_SERVER_URL}?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
            
            logger.info(f"Requesting audio from server: {url[:80]}...")
            
            # Download the audio
            response = requests.get(url, stream=True, timeout=20)
            
            if response.status_code != 200:
                logger.error(f"Server returned error status: {response.status_code}")
                self._is_playing = False
                self._playback_finished_event.set()
                return
            
            # Save to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                temp_path = temp_file.name
                
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        temp_file.write(chunk)
            
            # Check file size
            file_size = os.path.getsize(temp_path)
            logger.info(f"Downloaded audio file: {temp_path} ({file_size} bytes)")
            
            if file_size < 1000:  # Arbitrary check for valid audio
                logger.warning(f"Audio file is very small ({file_size} bytes)")
            
            # Play with ffplay
            logger.info("Playing audio with ffplay...")
            cmd = ["ffplay", "-autoexit", "-nodisp", temp_path]
            
            try:
                self._playback_process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                
                # Wait for ffplay to complete
                self._playback_process.wait()
                
                # Check return code
                if self._playback_process.returncode == 0:
                    logger.info("Audio played successfully!")
                else:
                    logger.error(f"ffplay returned code {self._playback_process.returncode}")
            except Exception as e:
                logger.error(f"Error during ffplay execution: {e}")
            
            # Clean up
            try:
                os.unlink(temp_path)
                logger.info(f"Temporary file removed: {temp_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file: {e}")
            
        except requests.RequestException as e:
            logger.error(f"Error requesting audio: {e}")
        except Exception as e:
            logger.error(f"Unexpected error during TTS processing: {e}")
        finally:
            # Reset state
            self._playback_process = None
            self._is_playing = False
            self._playback_finished_event.set()

    def say(self, text, voice=None, language=None):
        """
        Queue text to be spoken.
        
        Args:
            text: The text to speak
            voice: Optional voice ID (defaults to current setting)
            language: Optional language code (defaults to current setting)
            
        Returns:
            bool: True if successfully queued, False otherwise
        """
        if not FFPLAY_AVAILABLE:
            logger.error("Audio playback not available (ffplay not found)")
            return False
        
        if not text or not isinstance(text, str):
            logger.warning("Invalid text provided for TTS")
            return False
        
        # Clean the text
        cleaned_text = clean_text_for_tts(text)
        if not cleaned_text:
            logger.warning("Text was empty after cleaning")
            return False
        
        # Use defaults if not specified
        voice = voice or self.current_voice
        language = language or self.current_language
        
        # Add to queue
        with self._queue_lock:
            self._request_queue.append((cleaned_text, voice, language))
        
        logger.info(f"Added to queue: '{cleaned_text[:50]}...'")
        return True

    def stop_playback(self):
        """Stop any currently playing audio and clear the queue."""
        logger.info("Stopping all audio playback")
        
        # Clear the queue
        with self._queue_lock:
            self._request_queue.clear()
        
        # Stop current playback process
        if self._playback_process and self._is_playing:
            try:
                self._playback_process.terminate()
                time.sleep(0.5)
                if self._playback_process.poll() is None:
                    self._playback_process.kill()
            except Exception as e:
                logger.error(f"Error stopping playback: {e}")
        
        # Reset state
        self._is_playing = False
        self._playback_finished_event.set()
        logger.info("Audio playback stopped and queue cleared")

    def wait_until_safe_to_listen(self, timeout=30.0):
        """
        Wait until any queued TTS requests have finished playing.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            bool: True if safe to listen, False if timeout occurred
        """
        start_time = time.time()
        
        # First check if queue is empty
        while True:
            with self._queue_lock:
                queue_empty = not self._request_queue
            
            if queue_empty and not self._is_playing:
                return True
            
            # Check timeout
            if timeout is not None and (time.time() - start_time) > timeout:
                logger.warning(f"Timeout waiting for TTS to finish ({timeout}s)")
                return False
            
            # Wait for any current playback to finish
            if self._is_playing:
                remaining_time = timeout - (time.time() - start_time) if timeout else None
                logger.info(f"Waiting for current playback to finish (timeout={remaining_time}s)")
                result = self._playback_finished_event.wait(timeout=remaining_time)
                if not result:
                    logger.warning("Timeout waiting for playback to finish")
                    return False
            
            # Small sleep to prevent tight loop
            time.sleep(0.1)
    
    def set_voice(self, voice_id):
        """Set the default voice."""
        voice_exists = any(voice["id"] == voice_id for voice in AVAILABLE_VOICES)
        if voice_exists:
            self.current_voice = voice_id
            self._save_settings()
            logger.info(f"Default voice set to: {voice_id}")
            return True
        else:
            logger.warning(f"Voice '{voice_id}' not found in available voices")
            return False

    def set_language(self, language_id):
        """Set the default language."""
        language_exists = any(lang["id"] == language_id for lang in AVAILABLE_LANGUAGES)
        if language_exists:
            self.current_language = language_id
            self._save_settings()
            logger.info(f"Default language set to: {language_id}")
            return True
        else:
            logger.warning(f"Language '{language_id}' not found in available languages")
            return False
    
    def get_available_voices(self):
        """Return the list of available voices."""
        return AVAILABLE_VOICES
    
    def get_available_languages(self):
        """Return the list of available languages."""
        return AVAILABLE_LANGUAGES

# Create the singleton instance
_player_instance = StreamTTSPlayer()

# For testing
if __name__ == "__main__":
    print("StreamTTSPlayer Test")
    print("-----------------")
    
    if not FFPLAY_AVAILABLE:
        print("ffplay not available. Cannot run test.")
        sys.exit(1)
    
    print(f"Current voice: {_player_instance.current_voice}")
    print(f"Current language: {_player_instance.current_language}")
    
    # Test playback
    print("\nTesting TTS playback...")
    test_text = "This is a test of the streaming text to speech system using ffplay."
    success = _player_instance.say(test_text)
    
    if success:
        print("Playback queued. Waiting for completion...")
        finished = _player_instance.wait_until_safe_to_listen(timeout=15)
        if finished:
            print("Playback completed successfully!")
        else:
            print("Timeout waiting for playback to complete.")
    else:
        print("Failed to queue playback.")
    
    print("\nTest complete.")