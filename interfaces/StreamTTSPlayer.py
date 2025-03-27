# File: stream_tts.py
# --- REFACTOR: Simplified TTS playback using direct streaming ---
"""
Provides Text-to-Speech functionality by streaming audio directly from
an external TTS server and playing it using the sounddevice library.
Replaces the previous browser-based implementation.
"""

import requests
import sounddevice as sd
import threading
import time
import json
import os
import sys
import logging
import queue
from typing import Optional, Dict, Any, List
import numpy as np # Needed for sounddevice buffer conversion

# --- Refined Imports and Path Setup ---
# Assuming speech.py (for clean_text_for_tts) is in interfaces/ relative to project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import config # Assuming config.py is at the project root
    from interfaces import speech # For clean_text_for_tts
except ImportError as e:
    print(f"FATAL: Failed to import config or interfaces.speech: {e}", file=sys.stderr)
    print("Ensure config.py and interfaces/speech.py are accessible.", file=sys.stderr)
    sys.exit(1)

# --- Sounddevice Availability Check ---
try:
    sd.query_devices() # Check if sounddevice can find devices
    SOUNDDEVICE_AVAILABLE = True
    print("Sounddevice initialized successfully.")
except Exception as e:
    print(f"Warning: Failed to initialize sounddevice: {e}", file=sys.stderr)
    print("Audio playback will be disabled. Check audio device configuration and dependencies (like PortAudio).", file=sys.stderr)
    SOUNDDEVICE_AVAILABLE = False

# --- Logging Setup ---
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_file = os.path.join(log_dir, "stream_tts.log")

logger = logging.getLogger("stream_tts")
logger.setLevel(logging.INFO)
# Prevent duplicate logging if root logger is configured elsewhere
if not logger.hasHandlers():
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] - %(message)s'))
    logger.addHandler(file_handler)
    # Optional: Add console handler for debugging
    # console_handler = logging.StreamHandler(sys.stdout)
    # console_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    # logger.addHandler(console_handler)
logger.propagate = False # Prevent propagation to root logger

# --- Configuration ---
TTS_SERVER_URL = getattr(config, 'TTS_SERVER', "http://localhost:7851/api/tts-generate-streaming")
SETTINGS_FILE = "tts_settings.json"
DEFAULT_VOICE = "female_01.wav"
DEFAULT_LANGUAGE = "en"
# Audio stream parameters (NEEDS VERIFICATION from your TTS server documentation/output)
# Common values, adjust if your stream is different!
DEFAULT_SAMPLE_RATE = 24000 # E.g., 22050, 24000, 44100, 48000
DEFAULT_CHANNELS = 1 # Mono
DEFAULT_DTYPE = 'int16' # Assumes 16-bit PCM audio stream
STREAM_CHUNK_SIZE = 1024 # How many bytes to read from stream at a time

# Available voices/languages (keep hardcoded or load from config/API if dynamic)
AVAILABLE_VOICES = [
    {"id": "female_01.wav", "name": "Female 1"}, {"id": "female_02.wav", "name": "Female 2"},
    {"id": "female_03.wav", "name": "Female 3"}, {"id": "female_04.wav", "name": "Female 4"},
    {"id": "female_05.wav", "name": "Female 5"}, {"id": "male_01.wav", "name": "Male 1"},
    {"id": "male_02.wav", "name": "Male 2"}, {"id": "male_03.wav", "name": "Male 3"},
    {"id": "Morgan_Freeman CC3.wav", "name": "Morgan Freeman"} # Ensure this filename matches server
]
AVAILABLE_LANGUAGES = [
    {"id": "en", "name": "English"}, {"id": "fr", "name": "French"}, {"id": "es", "name": "Spanish"},
    {"id": "de", "name": "German"}, {"id": "it", "name": "Italian"}, {"id": "pt", "name": "Portuguese"},
    {"id": "nl", "name": "Dutch"}, {"id": "ru", "name": "Russian"}, {"id": "ja", "name": "Japanese"},
    {"id": "zh", "name": "Chinese"}
]

# --- Singleton TTS Player Class ---
class StreamTTSPlayer:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            logger.info("Creating StreamTTSPlayer instance.")
            cls._instance = super(StreamTTSPlayer, cls).__new__(cls)
            # --- Initialization ---
            cls._instance.current_voice = DEFAULT_VOICE
            cls._instance.current_language = DEFAULT_LANGUAGE
            # Playback state management
            cls._instance._playback_finished_event = threading.Event()
            cls._instance._playback_finished_event.set() # Initially set (nothing playing)
            cls._instance._current_stream_thread: Optional[threading.Thread] = None # type: ignore
            cls._instance._stop_playback_flag = threading.Event()
            cls._instance._output_stream: Optional[sd.OutputStream] = None # type: ignore
            # Load initial settings
            cls._instance._ensure_settings_file()
            cls._instance._load_settings()
            # Queue for handling sequential requests smoothly
            cls._instance._request_queue = queue.Queue()
            cls._instance._processing_thread = threading.Thread(target=cls._instance._process_queue, daemon=True)
            cls._instance._processing_thread.start()
        return cls._instance

    def _ensure_settings_file(self):
        """Make sure the settings file exists."""
        if not os.path.exists(SETTINGS_FILE):
            logger.info(f"Creating default settings file: {SETTINGS_FILE}")
            try:
                with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                    json.dump({
                        "default_voice": DEFAULT_VOICE,
                        "default_language": DEFAULT_LANGUAGE
                    }, f, indent=2)
            except IOError as e:
                logger.error(f"Failed to create settings file {SETTINGS_FILE}: {e}")

    def _load_settings(self):
        """Load settings from the settings file."""
        if not os.path.exists(SETTINGS_FILE):
            logger.warning(f"Settings file {SETTINGS_FILE} not found. Using defaults.")
            return
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                settings = json.load(f)
                self.current_voice = settings.get("default_voice", DEFAULT_VOICE)
                self.current_language = settings.get("default_language", DEFAULT_LANGUAGE)
            logger.info(f"Settings loaded. Voice: {self.current_voice}, Language: {self.current_language}")
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Error loading settings from {SETTINGS_FILE}: {e}. Using defaults.")
            self.current_voice = DEFAULT_VOICE
            self.current_language = DEFAULT_LANGUAGE

    def _save_settings(self):
        """Save current settings to the settings file."""
        try:
            with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
                json.dump({
                    "default_voice": self.current_voice,
                    "default_language": self.current_language
                }, f, indent=2)
            logger.info(f"Settings saved. Voice: {self.current_voice}, Language: {self.current_language}")
        except IOError as e:
            logger.error(f"Error saving settings to {SETTINGS_FILE}: {e}")

    def _playback_finished_callback(self):
        """Callback executed by sounddevice when the stream finishes."""
        logger.info("Sounddevice OutputStream finished.")
        if not self._playback_finished_event.is_set():
            self._playback_finished_event.set() # Signal completion

    def _stream_and_play(self, text: str, voice: str, language: str):
        """
        Internal method executed in a thread to stream and play TTS.
        Handles a single request.
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("Cannot play audio: sounddevice is not available.")
            self._playback_finished_event.set() # Signal completion immediately on error
            return

        self._stop_playback_flag.clear() # Ensure stop flag is cleared initially
        self._playback_finished_event.clear() # Signal that playback is starting
        logger.info(f"Starting playback for: '{text[:50]}...' (Voice: {voice}, Lang: {language})")

        # Prepare POST data (based on curl example)
        post_data = {
            "text": text,
            "voice": voice,
            "language": language,
            "output_file": f"stream_{time.time()}.wav" # Filename seems required by API?
        }

        stream = None # Initialize stream variable

        try:
            logger.debug(f"Sending POST request to {TTS_SERVER_URL}")
            response = requests.post(TTS_SERVER_URL, data=post_data, stream=True, timeout=30)
            response.raise_for_status() # Check for HTTP errors (4xx, 5xx)
            logger.debug("Request successful, starting stream processing.")

            # Assume audio parameters (Verify!)
            samplerate = DEFAULT_SAMPLE_RATE
            channels = DEFAULT_CHANNELS
            dtype = DEFAULT_DTYPE

            # Create sounddevice output stream
            # Use a callback=None stream and manually write data
            self._output_stream = sd.OutputStream(
                samplerate=samplerate,
                channels=channels,
                dtype=dtype,
                finished_callback=self._playback_finished_callback
            )
            self._output_stream.start()
            logger.debug("Sounddevice OutputStream started.")

            # Read stream and write to sounddevice
            for chunk in response.iter_content(chunk_size=STREAM_CHUNK_SIZE):
                if self._stop_playback_flag.is_set():
                    logger.info("Stop flag set, aborting playback.")
                    break
                if chunk:
                    # Assuming the chunk is raw PCM data matching dtype
                    # Need to convert bytes to NumPy array for sounddevice
                    try:
                        audio_data = np.frombuffer(chunk, dtype=dtype)
                        self._output_stream.write(audio_data)
                    except ValueError as e:
                         logger.error(f"Error converting chunk to audio data (check dtype/chunk size?): {e}")
                         # Continue or break? Continue for now.
                    except Exception as write_e:
                        logger.error(f"Error writing to sounddevice stream: {write_e}")
                        break # Stop playback on write error


            logger.debug("Finished iterating over stream content.")

        except requests.exceptions.Timeout:
            logger.error(f"Timeout connecting to TTS server: {TTS_SERVER_URL}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Error requesting TTS stream from {TTS_SERVER_URL}: {e}")
        except sd.PortAudioError as e:
             logger.error(f"Sounddevice/PortAudio Error: {e}. Check audio device settings.")
        except Exception as e:
            logger.error(f"Unexpected error during streaming/playback: {e}", exc_info=True)
        finally:
            # Ensure stream is stopped and resources are released
            if self._output_stream:
                try:
                    if not self._output_stream.closed:
                        # If stop wasn't called, wait for buffer to finish? Or just abort? Abort for quicker response.
                        self._output_stream.abort() # Abort immediately
                        self._output_stream.close()
                        logger.debug("Sounddevice OutputStream closed.")
                except Exception as close_e:
                    logger.error(f"Error closing sounddevice stream: {close_e}")
                finally:
                     self._output_stream = None

            # Ensure the finished event is set, even if errors occurred
            if not self._playback_finished_event.is_set():
                logger.warning("Playback finished (or errored out), setting finished event.")
                self._playback_finished_event.set()
            self._stop_playback_flag.clear() # Reset stop flag

    def _process_queue(self):
        """Worker thread function to process TTS requests from the queue."""
        while True:
            try:
                # Wait indefinitely for an item from the queue
                text, voice, language = self._request_queue.get()
                logger.info("Processing item from queue.")
                # This call will block until streaming/playback finishes or errors out
                self._stream_and_play(text, voice, language)
                self._request_queue.task_done() # Mark task as complete
                logger.info("Finished processing queue item.")
            except Exception as e:
                 logger.error(f"Error in queue processing loop: {e}", exc_info=True)
                 # Avoid exiting the thread, just log and continue
                 time.sleep(1) # Prevent tight loop on continuous errors


    def say(self, text: str, voice: Optional[str] = None, language: Optional[str] = None) -> bool:
        """
        Adds text to the TTS playback queue. Returns immediately.
        Playback happens sequentially in a background thread.

        Args:
            text (str): Text to speak.
            voice (Optional[str]): Voice ID to use (defaults to current setting).
            language (Optional[str]): Language code (defaults to current setting).

        Returns:
            bool: True if added to queue successfully, False otherwise.
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.error("Cannot queue 'say' command: sounddevice not available.")
            return False
        if not isinstance(text, str) or not text.strip():
             logger.warning("Ignoring empty text for TTS.")
             return False

        # Use defaults if not provided
        voice = voice or self.current_voice
        language = language or self.current_language

        # Add request details to the queue
        try:
            self._request_queue.put((text, voice, language))
            logger.info(f"Added to queue: '{text[:50]}...'")
            return True
        except Exception as e:
            logger.error(f"Failed to add text to TTS queue: {e}", exc_info=True)
            return False

    def wait_until_safe_to_listen(self, timeout: Optional[float] = 30.0) -> bool:
        """
        Waits until the TTS queue is empty and the last playback has finished.

        Args:
            timeout (Optional[float]): Maximum time to wait in seconds. None waits indefinitely.

        Returns:
            bool: True if queue is empty and playback finished within timeout, False otherwise.
        """
        if not SOUNDDEVICE_AVAILABLE:
            logger.warning("wait_until_safe_to_listen called but sounddevice unavailable.")
            return True # Assume safe if no audio can play

        logger.info("Waiting for TTS queue and playback to finish...")
        start_time = time.monotonic()

        # 1. Wait for the queue to become empty
        while not self._request_queue.empty():
            if timeout is not None and (time.monotonic() - start_time) > timeout:
                logger.warning(f"Timeout waiting for TTS queue to empty after {timeout}s.")
                return False
            time.sleep(0.1) # Small sleep while checking queue

        # 2. Wait for the currently playing audio (if any) to finish
        remaining_timeout = None
        if timeout is not None:
             elapsed = time.monotonic() - start_time
             remaining_timeout = max(0, timeout - elapsed)

        logger.debug(f"Queue empty, now waiting for playback event (timeout={remaining_timeout}).")
        finished = self._playback_finished_event.wait(timeout=remaining_timeout)

        if finished:
             logger.info("TTS queue empty and playback finished. Safe to listen.")
        else:
             logger.warning(f"Timeout ({timeout}s) waiting for TTS playback to finish.")

        return finished

    def stop_playback(self):
        """Stops the currently playing TTS audio immediately."""
        logger.info("Received request to stop current playback.")
        # Clear the queue to prevent subsequent items from playing
        while not self._request_queue.empty():
            try:
                self._request_queue.get_nowait()
                self._request_queue.task_done()
            except queue.Empty:
                break
            except Exception as e:
                 logger.error(f"Error clearing queue during stop: {e}")
        logger.info("TTS queue cleared.")

        # Signal the streaming thread to stop
        self._stop_playback_flag.set()

        # Abort the sounddevice stream directly if it exists
        if self._output_stream and not self._output_stream.closed:
            try:
                logger.info("Aborting sounddevice stream...")
                self._output_stream.abort()
                self._output_stream.close() # Ensure callback fires if abort doesn't
            except Exception as e:
                 logger.error(f"Error aborting/closing sounddevice stream: {e}")
            finally:
                self._output_stream = None

        # Ensure the finished event gets set
        if not self._playback_finished_event.is_set():
             self._playback_finished_event.set()
        logger.info("Playback stop signaled.")


    def set_voice(self, voice_id: str) -> bool:
        """Set the default voice."""
        voice_exists = any(voice["id"] == voice_id for voice in AVAILABLE_VOICES)
        if voice_exists:
            self.current_voice = voice_id
            self._save_settings()
            logger.info(f"Default voice set to: {voice_id}")
            return True
        else:
            logger.warning(f"Voice '{voice_id}' not found in AVAILABLE_VOICES.")
            return False

    def set_language(self, language_id: str) -> bool:
        """Set the default language."""
        language_exists = any(language["id"] == language_id for language in AVAILABLE_LANGUAGES)
        if language_exists:
            self.current_language = language_id
            self._save_settings()
            logger.info(f"Default language set to: {language_id}")
            return True
        else:
            logger.warning(f"Language '{language_id}' not found in AVAILABLE_LANGUAGES.")
            return False

    def get_available_voices(self) -> List[Dict[str, str]]:
        """Return a list of available voices."""
        return AVAILABLE_VOICES

    def get_available_languages(self) -> List[Dict[str, str]]:
        """Return a list of available languages."""
        return AVAILABLE_LANGUAGES

    def release(self):
        """Clean up resources."""
        logger.info("Releasing StreamTTSPlayer resources...")
        self.stop_playback() # Ensure playback is stopped and queue is cleared
        # No specific thread joining needed if daemon=True, but ensure queue is empty?
        # Maybe signal processor thread to exit? Add an exit sentinel to queue?
        # For simplicity, rely on daemon thread exiting with main program for now.
        logger.info("StreamTTSPlayer resources released.")


# --- Wrapper Functions ---
_player_instance: Optional[StreamTTSPlayer] = None

def _get_player() -> Optional[StreamTTSPlayer]:
    """Gets the singleton player instance, handling initialization."""
    global _player_instance
    if not SOUNDDEVICE_AVAILABLE:
         # Log this only once
         if _player_instance is None:
              logger.critical("Cannot create TTS player: sounddevice is unavailable.")
         _player_instance = None # Ensure it's None
         return None

    if _player_instance is None:
        try:
            _player_instance = StreamTTSPlayer()
        except Exception as e:
            logger.critical(f"Failed to initialize StreamTTSPlayer: {e}", exc_info=True)
            _player_instance = None
    return _player_instance

def say(text: str, voice: Optional[str] = None, language: Optional[str] = None) -> bool:
    """
    Adds cleaned text to the TTS playback queue. Returns immediately.

    Args:
        text (str): Text to speak.
        voice (Optional[str]): Voice ID to use (defaults to current setting).
        language (Optional[str]): Language code (defaults to current setting).

    Returns:
        bool: True if added to queue successfully, False otherwise.
    """
    player = _get_player()
    if not player:
        return False

    try:
        # Clean the text before queueing
        cleaned_text = speech.clean_text_for_tts(text)
        if not cleaned_text:
             logger.warning("Text became empty after cleaning, not queueing.")
             return False
        return player.say(cleaned_text, voice, language)
    except Exception as e:
        logger.error(f"Error in say wrapper function: {e}", exc_info=True)
        return False

def set_voice(voice_id: str) -> bool:
    """Set the default voice."""
    player = _get_player()
    return player.set_voice(voice_id) if player else False

def set_language(language_id: str) -> bool:
    """Set the default language."""
    player = _get_player()
    return player.set_language(language_id) if player else False

def get_available_voices() -> List[Dict[str, str]]:
    """Return a list of available voices."""
    player = _get_player()
    return player.get_available_voices() if player else []

def get_available_languages() -> List[Dict[str, str]]:
    """Return a list of available languages."""
    player = _get_player()
    return player.get_available_languages() if player else []

def wait_until_safe_to_listen(timeout: Optional[float] = 30.0) -> bool:
    """Waits until the TTS queue is empty and the last playback has finished."""
    player = _get_player()
    return player.wait_until_safe_to_listen(timeout) if player else True # Assume safe if no player

def stop_playback():
    """Stops the currently playing audio and clears the queue."""
    player = _get_player()
    if player:
        player.stop_playback()

def release_tts():
     """Releases resources used by the TTS player."""
     player = _get_player()
     if player:
          player.release()


# --- Example Usage ---
if __name__ == "__main__":
    print("StreamTTS Test")
    print("--------------")

    if not SOUNDDEVICE_AVAILABLE:
        print("Sounddevice not available, cannot run test.")

    else:
        # List available voices/languages
        print("\nAvailable voices:")
        for v in get_available_voices(): print(f"- {v['name']} ({v['id']})")
        print("\nAvailable languages:")
        for l in get_available_languages(): print(f"- {l['name']} ({l['id']})")

        # Load settings
        _player = _get_player()
        if _player:
            print(f"\nCurrent voice: {_player.current_voice}")
            print(f"Current language: {_player.current_language}")

            # Example calls
            say("Hello, this is a test of the simplified streaming TTS.")
            say("This text should play after the first one finishes.")
            say("Testing with a different voice.", voice="male_01.wav")
            say("Et maintenant, un peu de français.", voice="female_03.wav", language="fr") # Use a voice ID available on server

            print("\nText added to queue. Playback happening in background.")
            print("Waiting for playback to complete (max 30s)...")

            safe = wait_until_safe_to_listen(30)

            if safe:
                print("Playback finished.")
                say("Final test message.")
                print("Waiting for final message...")
                wait_until_safe_to_listen(10)
                print("Final message likely finished.")
            else:
                print("Timeout waiting for playback to finish.")
                stop_playback() # Stop if timed out

            print("Releasing TTS resources...")
            release_tts()
            print("TTS released.")
        else:
            print("Failed to initialize TTS Player.")