"""
Configuration settings for the Smart Assistant application.

This file centralizes settings like API keys, model parameters, prompts,
file paths, and service URLs used throughout the application.
"""

import os
from dotenv import load_dotenv
import logging  # Added logging level

# --- Load environment variables from .env file ---
load_dotenv()

# --- Logging Configuration ---
# Define log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO').upper()

# --- API Keys ---
GEMINI_KEY = os.getenv("AIzaSyAG8syGBFWBfgy8VmDcK5JhsBUDYHWsHss", "AIzaSyAG8syGBFWBfgy8VmDcK5JhsBUDYHWsHss") # Replace with your actual key if not using .env
VOICE_API_KEY = os.getenv("VOICE_API_KEY", "YOUR_DEFAULT_ELEVENLABS_KEY")  # Replace if needed
AWS_ACCESS = os.getenv("AWS_ACCESS", "YOUR_DEFAULT_AWS_ACCESS_KEY")  # Replace if needed
AWS_SECRET = os.getenv("AWS_SECRET", "YOUR_DEFAULT_AWS_SECRET_KEY")  # Replace if needed
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "YOUR_DEFAULT_WEATHERAPI_KEY")  # Default provided in original code
SPOTIPY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', 'YOUR_SPOTIFY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', 'YOUR_SPOTIFY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8888/callback')

# --- LLM Configuration ---
MODEL_NAME = "Cypher"
TEMPERATURE = 0.72
TOP_P = 0.95
TOP_K = 40
MAX_OUTPUT_TOKENS = 8192

SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Prompts ---
# SYSTEM_PROMPT remains the same as you provided before
SYSTEM_PROMPT = """You are Cypher, an advanced AI assistant integrated into a smart home environment.
You have access to various services including weather information, time telling, mood setting (via YouTube),
and potentially controlling Spotify (if configured). You also have a vision system that can be activated
to track a specific person named Felix.

Personality: You are helpful, slightly informal, knowledgeable, and proactive when appropriate.
You should clearly state when you are performing an action (like getting weather or starting video).

Vision System Interaction:

The vision system ("IseeYou") is NOT always active.

The user must explicitly ask you to start or stop video tracking (e.g., "start tracking Felix", "show me the camera", "stop video").

When asked to start, confirm you are starting the video feed.

When asked to stop, confirm you are stopping the video feed.

Do NOT assume the video is running unless the user has asked to start it.

You might receive information from the vision system about whether Felix is detected, but you don't control the camera directly, only the activation/deactivation of the tracking process.

Service Interaction:

Clearly indicate when you need to use a service (e.g., "Okay, I'll check the weather for you in [location].").

If you need information (like location for weather), ask the user.

Use the provided function triggers in your response when you intend to use a service. Use these specific phrases:

Weather: "FUNCTION_TRIGGER:GET_WEATHER:[LOCATION]" (replace [LOCATION] if specified, otherwise use default or ask)
Time: "FUNCTION_TRIGGER:TELL_TIME"
Mood Setter (YouTube): "FUNCTION_TRIGGER:SET_MOOD"
Start Video Tracking: "FUNCTION_TRIGGER:START_VIDEO"
Stop Video Tracking: "FUNCTION_TRIGGER:STOP_VIDEO"
Spotify (Example): "FUNCTION_TRIGGER:PLAY_SPOTIFY:[SONG/ARTIST/PLAYLIST]"

Example Interaction (Video):
User: Can you see if Felix is around?
Ma_Boi: I can start the video tracking to check. Should I begin?
User: Yes, please.
Ma_Boi: Okay, starting the video feed now to look for Felix. FUNCTION_TRIGGER:START_VIDEO
(Later)
User: Okay, you can stop the video now.
Ma_Boi: Alright, stopping the video tracking feed. FUNCTION_TRIGGER:STOP_VIDEO

Example Interaction (Weather):
User: What's the weather like today?
Ma_Boi: I can check that for you. Where are you located?
User: Seattle.
Ma_Boi: Okay, checking the weather for Seattle. FUNCTION_TRIGGER:GET_WEATHER:Seattle

Respond concisely and helpfully based on the user's query and the conversation history.
"""
time_prompt = "Include the current date and time in your response."
weather_prompt = "Provide a weather update based on this data: "
switch_mode_prompt = "The user wants to set a relaxing mood. Briefly introduce the YouTube video you are about to play (e.g., a calming nature scene or music) and then say 'FUNCTION_TRIGGER:SET_MOOD'."
switch_mode_prompt_2 = "Continue the mood setting. Generate a short, calming follow-up message or observation related to the mood."
start_video_prompt = "Confirm that you are starting the video tracking system to look for Felix. Then include the trigger: FUNCTION_TRIGGER:START_VIDEO"
stop_video_prompt = "Confirm that you are stopping the video tracking system. Then include the trigger: FUNCTION_TRIGGER:STOP_VIDEO"

# --- Service Configuration ---
DEFAULT_WEATHER_LOCATION = "Seattle"
YOUTUBE_MOOD_URL = "https://www.youtube.com/watch?v=ztVV54sPOns&t=461s"
NUMBER_STORIES = 3

# --- Vision System (IseeYou & GPU Server) ---
FELIX_SERVER_URL = os.getenv("FELIX_SERVER_URL", "ws://localhost:8765")
FELIX_VIDEO_SOURCE = os.getenv("FELIX_VIDEO_SOURCE", "0")  # Can be "0", "1", etc. for webcam, or a file/URL
FELIX_RECOGNIZER_THRESHOLD = 0.6
PERSON_DETECTION_THRESHOLD = 0.6
FELIX_MODEL_PATH = os.getenv("FELIX_MODEL_PATH", "/root/models/felix_classifier.pth")
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", None) # Specify path if using local YOLO model

# --- Whisper API ---
WHISPER_API_URL = os.getenv("WHISPER_API_URL", "http://localhost:5001/transcribe")

# Add STT Method choice here if desired
DEFAULT_STT_METHOD = os.getenv("DEFAULT_STT_METHOD", "whisper_api")  # Example

# --- TTS Configuration ---
DEFAULT_TTS_ENGINE = os.getenv("DEFAULT_TTS_ENGINE", "alltalk_tts")  # e.g., 'pyttsx3', 'google', 'elevenlab', 'edge', 'aws', 'alltalk_tts'
PYTTSX3_RATE = 150
ELEVENLABS_VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "pFZP5JQG7iQjIQuC4Bku")  # Example
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
EDGE_TTS_VOICE = "en-GB-SoniaNeural"  # Example
AWS_POLLY_VOICE_ID = "Amy"  # Example
AWS_POLLY_REGION = os.getenv("AWS_POLLY_REGION", "us-east-1")

# AllTalk TTS Server URL (used by StreamTTSPlayer)
TTS_SERVER = os.getenv("TTS_SERVER", "http://localhost:7851/api/tts-generate-streaming")

# --- NEW: UI Configuration ---
UI_WEBSOCKET_PORT = int(os.getenv("UI_WEBSOCKET_PORT", 8766))
UI_HTTP_PORT = int(os.getenv("UI_HTTP_PORT", 8080))  # Port for serving index.html etc.
UI_ENABLED_BY_DEFAULT = os.getenv("UI_ENABLED_BY_DEFAULT", "False").lower() in ('true', '1', 't') # Set True to skip prompt

# --- Input Validation ---
if not GEMINI_KEY or "YOUR_DEFAULT" in GEMINI_KEY:
    print("WARNING: GEMINI_KEY is not set or using default in config.py. LLM calls will likely fail.")
if DEFAULT_TTS_ENGINE == 'elevenlab' and (not VOICE_API_KEY or "YOUR_DEFAULT" in VOICE_API_KEY):
    print("WARNING: ElevenLabs TTS selected but VOICE_API_KEY is not set.")
if DEFAULT_TTS_ENGINE == 'aws' and (not AWS_ACCESS or "YOUR_DEFAULT" in AWS_ACCESS or not AWS_SECRET or "YOUR_DEFAULT" in AWS_SECRET):
    print("WARNING: AWS TTS selected but AWS_ACCESS or AWS_SECRET is not set.")

print(f"Configuration loaded. Log Level: {LOG_LEVEL}, TTS Engine: {DEFAULT_TTS_ENGINE}")
print(f"UI WebSocket Port: {UI_WEBSOCKET_PORT}, UI HTTP Port: {UI_HTTP_PORT}")