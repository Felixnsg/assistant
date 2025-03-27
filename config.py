# File: config.py
# --- REFACTOR: Added file description ---
"""
Configuration settings for the Smart Assistant application.

This file centralizes settings like API keys, model parameters, prompts,
file paths, and service URLs used throughout the application.
"""

import os
from dotenv import load_dotenv

# --- REFACTOR: Load environment variables from .env file ---
# Create a .env file in the same directory as config.py
# Example .env content:
# GEMINI_KEY=your_gemini_api_key
# VOICE_API_KEY=your_elevenlabs_api_key
# AWS_ACCESS=your_aws_access_key_id
# AWS_SECRET=your_aws_secret_access_key
# WEATHER_API_KEY=your_weatherapi_key
# SPOTIFY_CLIENT_ID=your_spotify_client_id
# SPOTIFY_CLIENT_SECRET=your_spotify_client_secret
# SPOTIFY_REDIRECT_URI=your_spotify_redirect_uri

load_dotenv()

# --- API Keys ---
# --- REFACTOR: Use os.getenv for safer key retrieval ---
GEMINI_KEY = os.getenv("AIzaSyAG8syGBFWBfgy8VmDcK5JhsBUDYHWsHss", "AIzaSyAG8syGBFWBfgy8VmDcK5JhsBUDYHWsHss") # Replace with your actual key if not using .env
VOICE_API_KEY = os.getenv("sk_68de112f7119057d1ffeaac0e1b883a60c69339379b5cc9f", "YOUR_DEFAULT_ELEVENLABS_KEY_HERE") # Replace if needed
AWS_ACCESS = os.getenv("AKIAVIOZGCFOAOGFIJNW", "YOUR_DEFAULT_AWS_ACCESS_KEY") # Replace if needed
AWS_SECRET = os.getenv("HdAJ2NNvgV  HB1TyhKomoFB1I8qVRO+c1IVKxUQ/x", "YOUR_DEFAULT_AWS_SECRET_KEY") # Replace if needed
WEATHER_API_KEY = os.getenv("X9TALQUE94PNFC2PL37JTXRHV", "66ff43b4ee214eed86763359251603") # Default provided in original code
SPOTIPY_CLIENT_ID = os.getenv('SPOTIFY_CLIENT_ID', 'YOUR_SPOTIFY_CLIENT_ID')
SPOTIPY_CLIENT_SECRET = os.getenv('SPOTIFY_CLIENT_SECRET', 'YOUR_SPOTIFY_CLIENT_SECRET')
SPOTIPY_REDIRECT_URI = os.getenv('SPOTIFY_REDIRECT_URI', 'http://localhost:8888/callback') # Common default


# --- LLM Configuration ---
MODEL_NAME = "Cypher" # Name for the assistant in printouts
# --- REFACTOR: Added comments explaining parameters ---
# Controls randomness. Lower values are more deterministic, higher values more creative.
TEMPERATURE = 0.72
# Nucleus sampling: Considers the smallest set of tokens whose cumulative probability exceeds top_p.
TOP_P = 0.95
# Selects the top K most likely tokens at each step.
TOP_K = 40
# Maximum number of tokens to generate in the response.
MAX_OUTPUT_TOKENS = 8192 # Adjust as needed

# Safety settings for the LLM (Example structure for Google Generative AI)
# Refer to the specific LLM API documentation for valid categories and thresholds.
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

# --- Prompts ---
SYSTEM_PROMPT = """You are Cypher, an advanced AI assistant integrated into a smart home environment.
You have access to various services including weather information, time telling, mood setting (via YouTube),
and potentially controlling Spotify (if configured). You also have a vision system that can be activated
to track a specific person named Felix.

Personality: You are helpful, slightly informal, knowledgeable, and proactive when appropriate.
You should clearly state when you are performing an action (like getting weather or starting video).

Vision System Interaction:
- The vision system ("IseeYou") is NOT always active.
- The user must explicitly ask you to start or stop video tracking (e.g., "start tracking Felix", "show me the camera", "stop video").
- When asked to start, confirm you are starting the video feed.
- When asked to stop, confirm you are stopping the video feed.
- Do NOT assume the video is running unless the user has asked to start it.
- You might receive information from the vision system about whether Felix is detected, but you don't control the camera directly, only the activation/deactivation of the tracking process.

Service Interaction:
- Clearly indicate when you need to use a service (e.g., "Okay, I'll check the weather for you in [location].").
- If you need information (like location for weather), ask the user.
- Use the provided function triggers in your response when you intend to use a service. Use these specific phrases:
    - Weather: "FUNCTION_TRIGGER:GET_WEATHER:[LOCATION]" (replace [LOCATION] if specified, otherwise use default or ask)
    - Time: "FUNCTION_TRIGGER:TELL_TIME"
    - Mood Setter (YouTube): "FUNCTION_TRIGGER:SET_MOOD"
    - Start Video Tracking: "FUNCTION_TRIGGER:START_VIDEO"
    - Stop Video Tracking: "FUNCTION_TRIGGER:STOP_VIDEO"
    - Spotify (Example): "FUNCTION_TRIGGER:PLAY_SPOTIFY:[SONG/ARTIST/PLAYLIST]"

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
weather_prompt = "Provide a weather update based on this data: " # Weather data will be appended
switch_mode_prompt = "The user wants to set a relaxing mood. Briefly introduce the YouTube video you are about to play (e.g., a calming nature scene or music) and then say 'FUNCTION_TRIGGER:SET_MOOD'."
switch_mode_prompt_2 = "Continue the mood setting. Generate a short, calming follow-up message or observation related to the mood."
# --- REFACTOR: Added prompts for video control ---
start_video_prompt = "Confirm that you are starting the video tracking system to look for Felix. Then include the trigger: FUNCTION_TRIGGER:START_VIDEO"
stop_video_prompt = "Confirm that you are stopping the video tracking system. Then include the trigger: FUNCTION_TRIGGER:STOP_VIDEO"

# --- Service Configuration ---
DEFAULT_WEATHER_LOCATION = "Seattle"
YOUTUBE_MOOD_URL = "https://www.youtube.com/watch?v=ztVV54sPOns&t=461s"
NUMBER_STORIES = 3 # Used in the mood setter sequence in utilities

# --- Vision System (IseeYou & GPU Server) ---
FELIX_SERVER_URL = os.getenv("FELIX_SERVER_URL", "ws://localhost:8765")
# --- REFACTOR: Use getenv with a default for video source ---
FELIX_VIDEO_SOURCE = os.getenv("FELIX_VIDEO_SOURCE", "0") # Default to camera 0
FELIX_RECOGNIZER_THRESHOLD = 0.6 # Confidence threshold for classifying Felix
PERSON_DETECTION_THRESHOLD = 0.6 # Confidence threshold for detecting any person
FELIX_MODEL_PATH = os.getenv("FELIX_MODEL_PATH", "/root/models/felix_classifier.pth") # Or adjust default path
YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", None) # Use default YOLO (yolov8x.pt) if None or path not found

# --- Whisper API ---
WHISPER_API_URL = os.getenv("WHISPER_API_URL", "http://localhost:5001/transcribe")

# --- TTS Configuration ---
# --- REFACTOR: Added config for default TTS engine ---
DEFAULT_TTS_ENGINE = "alltalk_tts" # Options: 'pyttsx3', 'google', 'elevenlab', 'edge', 'aws'
PYTTSX3_RATE = 150
ELEVENLABS_VOICE_ID = "pFZP5JQG7iQjIQuC4Bku" # Example voice ID
ELEVENLABS_MODEL_ID = "eleven_multilingual_v2"
EDGE_TTS_VOICE = "en-GB-SoniaNeural"
AWS_POLLY_VOICE_ID = "Amy" # Example voice ID
AWS_POLLY_REGION = "us-east-1"


# --- Input Validation ---
# --- REFACTOR: Basic validation for critical keys ---
if not GEMINI_KEY or "YOUR_DEFAULT" in GEMINI_KEY:
    print("WARNING: GEMINI_KEY is not set or using default in config.py. LLM calls will likely fail.")
# Add similar checks for other critical keys if desired

print("Configuration loaded.")