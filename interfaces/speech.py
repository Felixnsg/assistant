# File: speech.py
"""
Provides text-to-speech (TTS) and speech-to-text (STT) functionalities.

Supports multiple TTS engines and includes options for local and cloud-based STT.
Handles text cleaning before synthesis and includes error handling for robustness.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from interfaces import StreamTTSPlayer
import torch
from typing import Optional
import json
import os
import pyttsx3
from gtts import gTTS, gTTSError
from playsound import playsound
import speech_recognition as sr # Renamed import for clarity
import edge_tts
import asyncio
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import boto3
from botocore.exceptions import BotoCoreError, ClientError
import time
import re
import whisper # Used for model loading if doing local Whisper STT
import numpy as np # Used if doing local Whisper STT with audio data manipulation
import tempfile
import queue # Used for continuous whisper implementation
import threading # Used for continuous whisper implementation
import requests # Used for Whisper API STT
try:
    import config
except ImportError as e:
    print(f"Error importing config module in speech.py: {e}. Ensure config.py exists.")
    sys.exit(1)

# --- REFACTOR: Initialize pyttsx3 engine safely ---
try:
    engine = pyttsx3.init()
    # Set properties only if engine initialized successfully
    engine.setProperty("rate", config.PYTTSX3_RATE)
except Exception as e:
    print(f"Warning: Failed to initialize pyttsx3 engine: {e}. pyttsx3 TTS will not work.")
    engine = None

TTS_OUTPUT_MP3 = "tts_output.mp3"
STT_TEMP_WAV = "stt_input.wav" # Used if saving temp file locally

StreamTTSPlayer_instance = StreamTTSPlayer.StreamTTSPlayer()



# === Text Cleaning ===

def clean_text_for_tts(text: str) -> str:
    """
    Cleans text for better compatibility with TTS engines.

    Removes common Markdown symbols (*, _, `), list markers, excessive whitespace,
    and other characters that might be mispronounced or cause errors.

    Args:
        text (str): The input text to clean.

    Returns:
        str: The cleaned text.
    """
    if not isinstance(text, str):
        print("Warning: clean_text_for_tts received non-string input. Returning empty string.")
        return ""

    # Remove bold markdown (**text**)
    text = text.replace('**', '')
    # Remove italic markdown (*text* or _text_)
    text = text.replace('*', '')
    text = text.replace('_', '') # Handles single underscore emphasis too
    # Remove code blocks (```text```) and inline code (`text`)
    text = text.replace('```', '')
    text = text.replace('`', '')
    # Remove strikethrough (~~text~~)
    text = text.replace('~~', '')
    # Handle lists: remove leading hyphens/asterisks and maybe add pauses
    text = re.sub(r'^\s*[-*+]\s+', '', text, flags=re.MULTILINE) # Remove bullets at line start
    # Replace markdown links [text](url) with just the text part
    text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
    # Remove HTML tags (simple removal)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove specific problematic characters (adjust as needed)
    text = text.replace('#', '') # Remove markdown headers
    # Replace multiple spaces/newlines with a single space
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

# === TTS Engines ===

def pyttsx3_text_to_speech(ai_response: str) -> bool:
    """
    Basic TTS using the pyttsx3 library (offline).

    Args:
        ai_response (str): The text to synthesize.

    Returns:
        bool: True if successful, False otherwise.
    """
    global engine # Access the globally initialized engine
    if not engine:
        print("Error: pyttsx3 engine not initialized. Cannot perform TTS.")
        return False
    if not isinstance(ai_response, str) or not ai_response.strip():
        print("Error: No text provided for pyttsx3 TTS.")
        return False

    try:
        cleaned_text = clean_text_for_tts(ai_response)
        print(f"pyttsx3 Speaking: {cleaned_text[:100]}...") # Log start
        # Ensure engine properties are set (might reset sometimes)
        engine.setProperty("rate", config.PYTTSX3_RATE)
        engine.say(cleaned_text)
        engine.runAndWait()
        print("pyttsx3 Finished Speaking.")
        return True
    except RuntimeError as e:
         print(f"Error during pyttsx3 runAndWait: {e}. Engine might be busy or invalid state.")
         # Attempt to re-initialize engine? Or just report failure.
         engine = None # Mark engine as potentially broken
         try: # Try re-init
              engine = pyttsx3.init()
              engine.setProperty("rate", config.PYTTSX3_RATE)
         except Exception as reinit_e:
              print(f"Failed to re-initialize pyttsx3 engine: {reinit_e}")
              engine = None
         return False
    except Exception as e:
        print(f"Error in pyttsx3_text_to_speech: {e}")
        return False

def google_tts_text_to_speech(ai_response: str, output_file: str = TTS_OUTPUT_MP3) -> bool:
    """
    Google Text-to-Speech using gTTS library (requires internet).

    Saves the speech to an MP3 file and plays it using playsound.

    Args:
        ai_response (str): The text to synthesize.
        output_file (str): Path to save the generated MP3 file.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not isinstance(ai_response, str) or not ai_response.strip():
        print("Error: No text provided for Google TTS.")
        return False

    try:
        cleaned_text = clean_text_for_tts(ai_response)
        print(f"Google TTS Synthesizing: {cleaned_text[:100]}...")

        language = 'en' # I am Considering making this more configurable, but since I don't really use it, doesn't worth the hussle.
        try:
            ai_speech = gTTS(text=cleaned_text, lang=language, slow=False)
        except gTTSError as e:
            print(f"gTTS Error: {e}. Check internet connection or text content.")
            return False
        except Exception as e: # Catch other potential gTTS init errors
             print(f"Unexpected error initializing gTTS: {e}")
             return False

        try:
            if os.path.exists(output_file):
                os.remove(output_file)
            ai_speech.save(output_file)
        except PermissionError:
            print(f"Error: Permission denied saving TTS file to '{output_file}'.")
            return False
        except Exception as e:
            print(f"Error saving gTTS output to '{output_file}': {e}")
            return False

        print(f"Google TTS Playing '{output_file}'...")
        try:
            playsound(output_file)
            print("Google TTS Finished Playing.")
            return True
        except Exception as e: # playsound can have various backend issues
            print(f"Error playing sound file '{output_file}' with playsound: {e}")
            return False
        finally:
             try:
                 if os.path.exists(output_file):
                     os.remove(output_file) # Clean up the temp file
             except OSError as e:
                 print(f"Warning: Could not remove temporary TTS file '{output_file}': {e}")

    except Exception as e:
        print(f"An unexpected error occurred in google_tts_text_to_speech: {e}")
        return False

def elevenlabs_tts_text_to_speech(ai_response: str) -> bool:
    """
    Premium TTS using the ElevenLabs API (requires API key and internet).

    Streams the audio directly for playback.

    Args:
        ai_response (str): The text to synthesize.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not config.VOICE_API_KEY or "YOUR_DEFAULT" in config.VOICE_API_KEY:
        print("Error: ElevenLabs API key not configured.")
        return False
    if not isinstance(ai_response, str) or not ai_response.strip():
        print("Error: No text provided for ElevenLabs TTS.")
        return False

    try:
        cleaned_text = clean_text_for_tts(ai_response)
        print(f"ElevenLabs TTS Synthesizing: {cleaned_text[:100]}...")

        client = ElevenLabs(api_key=config.VOICE_API_KEY)
        audio_stream = client.text_to_speech.convert_as_stream(
            text=cleaned_text,
            voice_id=config.ELEVENLABS_VOICE_ID,
            model_id=config.ELEVENLABS_MODEL_ID
        )
        print("ElevenLabs TTS Streaming...")
        stream(audio_stream) # This blocks until streaming is done
        print("ElevenLabs TTS Finished Streaming.")
        return True
    except requests.exceptions.RequestException as e:
         print(f"Network error during ElevenLabs request: {e}")
         return False
    except Exception as e:
        print(f"An unexpected error occurred in elevenlabs_tts_text_to_speech: {e}")
        return False

async def _edge_tts_save_to_mp3(text: str, filename: str = TTS_OUTPUT_MP3) -> Optional[str]:
    """
    Async helper to synthesize text using Edge TTS and save to an MP3 file.

    Args:
        text (str): The cleaned text to synthesize.
        filename (str): Path to save the MP3 file.

    Returns:
        Optional[str]: The filename if successful, None otherwise.
    """
    try:
        communicate = edge_tts.Communicate(text, config.EDGE_TTS_VOICE)
        try:
            if os.path.exists(filename):
                os.remove(filename)
            await communicate.save(filename)
            return filename
        except PermissionError:
             print(f"Error: Permission denied saving Edge TTS file to '{filename}'.")
             return None
        except Exception as e: # Catch edge_tts specific save errors if any documented
             print(f"Error saving Edge TTS output to '{filename}': {e}")
             return None
    except Exception as e:
        print(f"Error during Edge TTS communication/synthesis: {e}")
        return None

# --- REFACTOR: Renamed, added type hinting, improved error handling ---
def edge_tts_text_to_speech(ai_response: str, output_file: str = TTS_OUTPUT_MP3) -> bool:
    """
    Microsoft Edge TTS using the edge-tts library (requires internet).

    Saves speech to an MP3 file and plays it.

    Args:
        ai_response (str): The text to synthesize.
        output_file (str): Path to save the generated MP3 file.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not isinstance(ai_response, str) or not ai_response.strip():
        print("Error: No text provided for Edge TTS.")
        return False

    try:
        cleaned_text = clean_text_for_tts(ai_response)
        print(f"Edge TTS Synthesizing: {cleaned_text[:100]}...")

        try:
            # Get existing loop or create a new one if needed
            loop = asyncio.get_event_loop()
            if loop.is_running():
                filename = asyncio.run(_edge_tts_save_to_mp3(cleaned_text, output_file))
            else:
                filename = loop.run_until_complete(_edge_tts_save_to_mp3(cleaned_text, output_file))

        except RuntimeError as e:
            print(f"Asyncio loop error: {e}. Trying asyncio.run directly.")
            try:
                filename = asyncio.run(_edge_tts_save_to_mp3(cleaned_text, output_file))
            except Exception as run_e:
                print(f"Failed to run Edge TTS synthesis: {run_e}")
                return False
        except Exception as e: # Catch errors during async execution
            print(f"Error running Edge TTS synthesis task: {e}")
            return False


        if filename and os.path.exists(filename):
            print(f"Edge TTS Playing '{filename}'...")
            try:
                playsound(filename)
                print("Edge TTS Finished Playing.")
                return True
            except Exception as e:
                print(f"Error playing sound file '{filename}' with playsound: {e}")
                return False
            finally:
                # Clean up the temp file
                try:
                    if os.path.exists(filename):
                        os.remove(filename)
                except OSError as e:
                    print(f"Warning: Could not remove temporary TTS file '{filename}': {e}")
        else:
            print("Edge TTS failed to generate audio file.")
            return False

    except Exception as e:
        print(f"An unexpected error occurred in edge_tts_text_to_speech: {e}")
        return False

def aws_polly_text_to_speech(ai_response: str, output_file: str = TTS_OUTPUT_MP3) -> bool:
    """
    AWS Polly Text-to-Speech (requires AWS credentials and internet).

    Saves speech to an MP3 file and plays it.

    Args:
        ai_response (str): The text to synthesize.
        output_file (str): Path to save the generated MP3 file.

    Returns:
        bool: True if successful, False otherwise.
    """
    if not config.AWS_ACCESS or "YOUR_DEFAULT" in config.AWS_ACCESS or \
       not config.AWS_SECRET or "YOUR_DEFAULT" in config.AWS_SECRET:
        print("Error: AWS credentials (Access Key ID and Secret Key) not configured.")
        return False
    if not isinstance(ai_response, str) or not ai_response.strip():
        print("Error: No text provided for AWS Polly TTS.")
        return False

    try:
        cleaned_text = clean_text_for_tts(ai_response)
        print(f"AWS Polly Synthesizing: {cleaned_text[:100]}...")

        try:
            polly_client = boto3.client(
                "polly",
                region_name=config.AWS_POLLY_REGION,
                aws_access_key_id=config.AWS_ACCESS,
                aws_secret_access_key=config.AWS_SECRET
            )
        except (BotoCoreError, ClientError) as e:
            print(f"Error initializing AWS Polly client: {e}. Check credentials and region.")
            return False
        except Exception as e: # Catch other potential init errors
            print(f"Unexpected error initializing AWS Polly client: {e}")
            return False

        try:
            response = polly_client.synthesize_speech(
                Text=cleaned_text,
                VoiceId=config.AWS_POLLY_VOICE_ID,
                OutputFormat="mp3",
                Engine="neural" # Use neural engine for better quality
            )
        except (BotoCoreError, ClientError) as e:
             print(f"AWS Polly API Error: {e}. Check request parameters, permissions, or text length limits.")
             return False
        except Exception as e: # Catch other potential API errors
            print(f"Unexpected error calling Polly synthesize_speech: {e}")
            return False

        try:
            if os.path.exists(output_file):
                os.remove(output_file)

            # Check if audio stream exists
            if "AudioStream" in response:
                with open(output_file, "wb") as file:
                    file.write(response["AudioStream"].read())
            else:
                print("Error: No AudioStream found in AWS Polly response.")
                return False
        except PermissionError:
             print(f"Error: Permission denied saving Polly TTS file to '{output_file}'.")
             return False
        except IOError as e:
             print(f"Error writing Polly TTS output to '{output_file}': {e}")
             return False
        except Exception as e:
            print(f"Error processing Polly AudioStream or saving file: {e}")
            return False

        print(f"AWS Polly Playing '{output_file}'...")
        try:
            playsound(output_file)
            print("AWS Polly Finished Playing.")
            return True
        except Exception as e:
            print(f"Error playing sound file '{output_file}' with playsound: {e}")
            return False
        finally:
            # Clean up the temp file
            try:
                if os.path.exists(output_file):
                    os.remove(output_file)
            except OSError as e:
                print(f"Warning: Could not remove temporary TTS file '{output_file}': {e}")

    except Exception as e:
        print(f"An unexpected error occurred in aws_polly_text_to_speech: {e}")
        return False

# === Master TTS Function ===

def text_to_speech(ai_response: str, engine_choice: str = config.DEFAULT_TTS_ENGINE) -> bool:
    """
    Synthesizes text to speech using the configured or specified engine.

    Args:
        ai_response (str): The text to speak.
        engine_choice (str): The TTS engine to use. Defaults to config.DEFAULT_TTS_ENGINE.
                             Options: 'pyttsx3', 'google', 'elevenlab', 'edge', 'aws'.

    Returns:
        bool: True if TTS was successful, False otherwise.
    """
    print(f"--- Attempting TTS using engine: {engine_choice} ---")
    if engine_choice == 'pyttsx3':
        return pyttsx3_text_to_speech(ai_response)
    elif engine_choice == 'google':
        return google_tts_text_to_speech(ai_response)
    elif engine_choice == 'elevenlab':
        return elevenlabs_tts_text_to_speech(ai_response)
    elif engine_choice == 'edge':
        return edge_tts_text_to_speech(ai_response)
    elif engine_choice == 'aws':
        return aws_polly_text_to_speech(ai_response)
    elif engine_choice == 'alltalk_tts':
        return StreamTTSPlayer_instance.say()
    else:
        print(f"Error: Unknown TTS engine choice '{engine_choice}'. Falling back to pyttsx3 (if available).")
        return pyttsx3_text_to_speech(ai_response)


# === STT Engines ===

def google_speech_recognition() -> str:
    """
    Performs speech recognition using Google's Web Speech API via SpeechRecognition library.

    Listens via the default microphone, adjusts for ambient noise, and transcribes.

    Returns:
        str: The recognized text in lowercase, or an empty string if recognition fails
             or no speech is detected.
    """
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000  # Consider making configurable
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 0.8 # Default, seconds of silence indicating phrase end

    with sr.Microphone() as mic:
        print("Adjusting for ambient noise... Please wait.")
        try:
            # --- REFACTOR: Use context manager for adjustment ---
            recognizer.adjust_for_ambient_noise(mic, duration=1.0)
            print("Ambient noise adjustment complete.")
        except Exception as e:
            print(f"Warning: Failed to adjust for ambient noise: {e}")
            # Continue anyway, recognition might be less accurate

        print("Listening for command...")
        try:
            audio = recognizer.listen(mic, timeout=5, phrase_time_limit=15) 
            print("Audio captured, recognizing...")

            try:
                text = recognizer.recognize_google(audio)
                text = text.lower()
                print(f"Google STT Recognized: {text}")
                return text
            except sr.UnknownValueError:
                print("Google STT Error: Could not understand audio.")
                return ""
            except sr.RequestError as e:
                print(f"Google STT Error: Could not request results from Google Speech Recognition service; {e}. Check internet connection.")
                return ""
            except Exception as e: # Catch other potential recognition errors
                 print(f"An unexpected error occurred during Google STT recognition: {e}")
                 return ""

        except sr.WaitTimeoutError:
            print("Google STT Error: Listening timed out. No speech detected within the timeout period.")
            return ""
        except Exception as e: # Catch errors during listen()
             print(f"An error occurred during audio listening: {e}")
             return ""
    # except OSError as e:
         # print(f"Error accessing microphone: {e}. Ensure it's connected and permissions are granted.")
         # return "" # This needs to be handled carefully depending on sr.Microphone behavior
    # except Exception as e: # Catch potential sr.Microphone init errors
    #      print(f"Error initializing microphone: {e}")
    #      return ""


WHISPER_API_TIMEOUT = 30 # Timeout for API request in seconds

def whisper_api_speech_recognition() -> str:
    """
    Uses a remote Whisper API endpoint (Flask app) to transcribe audio from the microphone.

    Records audio, saves temporarily, sends to API, deletes file, returns transcription.

    Returns:
        str: The recognized text, or an empty string on failure.
    """
    recognizer = sr.Recognizer()
    # Configure recognizer parameters (optional, defaults might be okay)
    recognizer.energy_threshold = 3000 # Adjust based on testing
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 1.5 # Seconds of silence

    temp_file_path = None # Initialize path variable

    with sr.Microphone() as source:
        print("Adjusting for ambient noise (Whisper STT)... Please wait.")
        try:
            recognizer.adjust_for_ambient_noise(source, duration=1.0)
            print("Ambient noise adjustment complete.")
        except Exception as e:
            print(f"Warning: Failed to adjust for ambient noise: {e}")

        print("Listening for command (Whisper STT)...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=15)
            print("Audio captured, preparing for Whisper API...")
        except sr.WaitTimeoutError:
            print("Whisper STT Error: Listening timed out.")
            return ""
        except Exception as e:
            print(f"An error occurred during audio listening for Whisper: {e}")
            return ""

    try:
        # Use get_wav_data() which is standard
        audio_data = audio.get_wav_data()

        # Create a temporary file safely
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav', mode='wb') as temp_file:
            temp_file.write(audio_data)
            temp_file_path = temp_file.name
            print(f"Audio saved temporarily to: {temp_file_path}")

        api_url = config.WHISPER_API_URL
        print(f"Sending audio to Whisper API at {api_url}...")
        try:
            with open(temp_file_path, 'rb') as f:
                files = {'file': (os.path.basename(temp_file_path), f, 'audio/wav')}
                response = requests.post(api_url, files=files, timeout=WHISPER_API_TIMEOUT)

            if response.status_code == 200:
                try:
                    result = response.json()
                    if result.get('Result', '').lower() == 'success' and 'Text' in result:
                        transcription = result['Text']
                        print(f"Whisper API Recognized: {transcription}")
                        return transcription
                    elif result.get('Result', '').lower() == 'failure':
                         print(f"Whisper API Error: Transcription failed. Reason: {result.get('Text', 'Unknown')}")
                         return ""
                    else:
                         print(f"Whisper API Error: Unexpected JSON response format: {result}")
                         return ""
                except json.JSONDecodeError:
                    print(f"Whisper API Error: Could not decode JSON response. Response text: {response.text[:200]}...")
                    return ""
                except Exception as e: # Catch errors during JSON processing
                     print(f"Error processing Whisper API JSON response: {e}")
                     return ""
            else:
                print(f"Whisper API Error: Request failed with status code {response.status_code}.")
                print(f"Response: {response.text[:200]}...")
                return ""

        except requests.exceptions.Timeout:
            print(f"Whisper API Error: Request timed out after {WHISPER_API_TIMEOUT} seconds.")
            return ""
        except requests.exceptions.RequestException as e:
            print(f"Whisper API Error: Could not connect or send request to API at {api_url}. Error: {e}")
            return ""
        except Exception as e: # Catch other errors during API call
            print(f"An unexpected error occurred during Whisper API request: {e}")
            return ""

    except sr.exceptions.AudioDataError as e:
         print(f"Error getting WAV data from audio source: {e}")
         return ""
    except IOError as e:
        print(f"Error writing temporary audio file: {e}")
        return ""
    except Exception as e:
        print(f"An unexpected error occurred before or during Whisper API call setup: {e}")
        return ""
    finally:
        # --- REFACTOR: Ensure temporary file deletion ---
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
                print(f"Temporary file deleted: {temp_file_path}")
            except OSError as e:
                print(f"Warning: Could not delete temporary file '{temp_file_path}': {e}")


# === Continuous Whisper (Conceptual - Requires more integration) ===
# The continuous recorder/processor logic using threads/queues and local Whisper
# is more complex and tightly coupled. Let's keep the functions but note
# they need careful integration into the main app flow if used.

# --- REFACTOR: Keep continuous components but add notes ---
# Note: The following continuous Whisper components require careful integration
# into the main application loop (e.g., ChatManager or main.py) to manage
# the background thread and process the queue effectively without blocking.
# They are not directly used by the standard STT calls above.

# --- REFACTOR: Load Whisper model outside functions if used globally ---
# Consider loading the model once in main.py or ChatManager if using local continuous whisper
# whisper_model = None
# try:
#     # whisper_model = whisper.load_model("base") # Or choose model size
#     pass # Defer loading until needed / configured
# except Exception as e:
#     print(f"Warning: Failed to load local Whisper model: {e}. Local continuous STT may fail.")

audio_queue = queue.Queue()
stop_listening_event = threading.Event() # Renamed for clarity

def continuous_audio_recorder(stop_event: threading.Event, data_queue: queue.Queue):
    """
    Records audio continuously in a background thread and puts chunks into a queue.

    Args:
        stop_event (threading.Event): Event to signal when to stop recording.
        data_queue (queue.Queue): Queue to put recorded audio data into.
    """
    recognizer = sr.Recognizer()
    # --- REFACTOR: Use config/defaults for parameters ---
    recognizer.energy_threshold = 300 # Low threshold to catch more audio
    recognizer.dynamic_energy_threshold = True
    recognizer.pause_threshold = 2 # Longer pause indicates end of speech segment

    mic = None # Initialize
    try:
        mic = sr.Microphone()
        with mic as source:
            print("[Continuous Recorder] Adjusting for ambient noise...")
            try:
                 recognizer.adjust_for_ambient_noise(source, duration=1.0)
                 print("[Continuous Recorder] Noise adjustment complete. Starting continuous listening.")
            except Exception as e:
                 print(f"[Continuous Recorder] Warning: Noise adjustment failed: {e}")

            while not stop_event.is_set():
                try:
                    # Listen for a chunk of audio
                    audio = recognizer.listen(source, timeout=1, phrase_time_limit=15) # Shorter timeout, longer phrase limit
                    if audio:
                        data_queue.put(audio) # Put the audio data object in the queue
                        # print("[Continuous Recorder] Audio chunk added to queue.") # Debug log
                    # time.sleep(0.1) # Reduce sleep or remove if listen timeout handles waiting
                except sr.WaitTimeoutError:
                    # print("[Continuous Recorder] Timeout, listening again.") # Debug log
                    continue # No speech detected in timeout window, just loop again
                except Exception as e:
                    print(f"[Continuous Recorder] Error during listening: {e}")
                    # Decide if error is fatal
                    if not stop_event.is_set(): # Avoid printing error if stopping
                         print("[Continuous Recorder] Error is potentially critical, stopping.")
                         stop_event.set() # Signal stop on critical error
                    break # Exit loop on error
        print("[Continuous Recorder] Loop finished.")

    except OSError as e:
         print(f"[Continuous Recorder] Error accessing microphone: {e}. Thread stopping.")
         stop_event.set() # Signal stop
    except Exception as e:
         print(f"[Continuous Recorder] Unexpected error initializing or using microphone: {e}. Thread stopping.")
         stop_event.set() # Signal stop
    finally:
         print("[Continuous Recorder] Thread exiting.")


def process_audio_queue_with_whisper(stop_event: threading.Event, data_queue: queue.Queue, model_size="base"):
     """
     Processes audio chunks from a queue using a local Whisper model. (Conceptual)

     Args:
         stop_event (threading.Event): Event to signal when to stop processing.
         data_queue (queue.Queue): Queue to get audio data from.
         model_size (str): Whisper model size ('tiny', 'base', 'small', 'medium', 'large').
     """
     # --- REFACTOR: Load model inside the processing thread/function ---
     # This ensures the model is loaded in the thread where it will be used,
     # potentially important for GPU context.
     local_whisper_model = None
     try:
         print(f"[Whisper Processor] Loading Whisper model '{model_size}'...")
         local_whisper_model = whisper.load_model(model_size)
         print("[Whisper Processor] Model loaded.")
     except Exception as e:
         print(f"[Whisper Processor] FATAL: Failed to load local Whisper model '{model_size}': {e}")
         stop_event.set() # Signal other threads to stop if model fails
         return # Exit thread

     print("[Whisper Processor] Ready to process audio queue.")
     while not stop_event.is_set():
         try:
             # Wait for audio data with a timeout
             audio_data = data_queue.get(block=True, timeout=1)

             # Process the audio data
             # --- REFACTOR: Process directly from audio data if possible ---
             # Whisper might need a file path or a NumPy array. Convert audio_data.
             # Get raw data and sample rate/width
             raw_data = audio_data.get_raw_data()
             sample_rate = audio_data.sample_rate
             sample_width = audio_data.sample_width

             # Convert raw data to NumPy array (assuming 16-bit PCM)
             # Check sample_width to be sure! Whisper expects float32.
             if sample_width == 2: # 16-bit
                 audio_np = np.frombuffer(raw_data, dtype=np.int16).astype(np.float32) / 32768.0
             else:
                 print(f"[Whisper Processor] Warning: Unsupported sample width {sample_width}. Skipping chunk.")
                 continue

             if audio_np.size == 0:
                  print("[Whisper Processor] Warning: Empty audio chunk received. Skipping.")
                  continue

             print(f"[Whisper Processor] Transcribing audio chunk (duration: {len(audio_np)/sample_rate:.2f}s)...")
             result = local_whisper_model.transcribe(audio_np, fp16=torch.cuda.is_available()) # Use fp16 if GPU available
             text = result['text'].strip()

             if text: # Only process if transcription is not empty
                 print(f"[Whisper Processor] Recognized: {text}")
                 # --- TODO: Integrate this text back into ChatManager ---
                 # This part is crucial. How does this recognized text trigger
                 # a conversation turn in ChatManager? Needs a callback mechanism
                 # or another queue to send text back to the main thread/ChatManager.
                 # For now, just printing.
                 pass

             data_queue.task_done() # Mark task as complete

         except queue.Empty:
             # Queue was empty within the timeout, just loop again
             continue
         except Exception as e:
             if not stop_event.is_set(): # Avoid printing error if stopping
                 print(f"[Whisper Processor] Error during transcription: {e}")
             # Optionally add more specific error handling for Whisper
             data_queue.task_done() # Ensure task_done is called even on error
             # Decide if error is fatal and set stop_event

     print("[Whisper Processor] Processing loop finished.")


# --- Master STT Function ---
# --- REFACTOR: Added master STT function ---
def speech_to_text(method: str = "whisper_api") -> str:
    """
    Performs speech-to-text using the specified method.

    Args:
        method (str): The STT method to use. Options: 'google', 'whisper_api'.
                      'whisper_local_continuous' is conceptual and not fully integrated here.

    Returns:
        str: The recognized text, or an empty string on failure.
    """
    print(f"--- Attempting STT using method: {method} ---")
    if method == "google":
        return google_speech_recognition()
    elif method == "whisper_api":
        return whisper_api_speech_recognition()
    # elif method == "whisper_local_continuous":
    #     print("Warning: 'whisper_local_continuous' selected, but requires manual thread/queue management.")
    #     # This would need to start/manage the threads and get results, complex here.
    #     return "" # Placeholder
    else:
        print(f"Error: Unknown STT method '{method}'. Falling back to Whisper API.")
        return whisper_api_speech_recognition()