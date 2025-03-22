import os
import pyttsx3
from gtts import gTTS
from playsound import playsound
import speech_recognition
import edge_tts
import asyncio
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
import config
import boto3
import time
import re
import os
import speech_recognition as sr
import whisper
import numpy as np
import tempfile
from datetime import datetime, timedelta
from time import sleep

# Initialize the pyttsx3 engine globally
engine = pyttsx3.init()

def clean_text_for_tts(text):
    """
    Cleans text before sending to TTS engines by removing Markdown and other symbols
    that might cause issues with speech synthesis.
    """
    # Remove bold markdown
    text = text.replace('**', '')
    
    # Remove italic markdown
    text = text.replace('*', '')
    
    # Remove underline markdown
    text = text.replace('__', '')
    
    # Remove code blocks
    text = text.replace('```', '')
    text = text.replace('`', '')
    
    # Handle lists
    text = text.replace('- ', ', ')
    
    # Replace multiple spaces with single space
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def text_to_speech(ai_response):
    """Basic TTS using pyttsx3"""
    try:
        # Clean the text of markdown symbols
        cleaned_text = clean_text_for_tts(ai_response)
        
        engine.setProperty("rate", 150)
        engine.say(f"{cleaned_text}")
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"Error in text_to_speech: {e}")
        return False

def google_text_to_speech(ai_response):
    """Google Text-to-Speech using gTTS"""
    try:
        # Clean the text of markdown symbols
        cleaned_text = clean_text_for_tts(ai_response)
        
        language = 'en'
        ai_speech = gTTS(text=cleaned_text, lang=language, slow=False)
        output_file = "my_speech.mp3"
        
        if os.path.exists(output_file):
            os.remove(output_file)
            
        ai_speech.save(output_file)
        playsound(output_file)
        return True
    except Exception as e:
        print(f"Error in google_text_to_speech: {e}")
        return False

def elevenLab_text_to_speech(ai_response):
    """ElevenLabs premium TTS"""
    try:
        # Clean the text of markdown symbols
        cleaned_text = clean_text_for_tts(ai_response)
        
        client = ElevenLabs(api_key=config.VOICE_API_KEY)
        audio_stream = client.text_to_speech.convert_as_stream(
            text=cleaned_text,
            voice_id="pFZP5JQG7iQjIQuC4Bku",
            model_id="eleven_multilingual_v2"
        )
        # Play the streamed audio locally
        stream(audio_stream)
        return True
    except Exception as e:
        print(f"Error in elevenLab_text_to_speech: {e}")
        return False

async def text_to_mp3(ai_response, filename="response.mp3"):
    """Helper function for Edge TTS to save audio to file"""
    try:
        # Clean the text of markdown symbols
        cleaned_text = clean_text_for_tts(ai_response)
        
        communicate = edge_tts.Communicate(cleaned_text, "en-GB-SoniaNeural")
        if os.path.exists(filename):
            os.remove(filename)
        await communicate.save(filename)
        return filename
    except Exception as e:
        print(f"Error in text_to_mp3: {e}")
        return None

def edge_text_to_speech(ai_response):
    """Microsoft Edge TTS"""
    try:
        filename = "response.mp3"
        asyncio.run(text_to_mp3(ai_response, filename))
        if os.path.exists(filename):
            playsound(filename)
            return True
        return False
    except Exception as e:
        print(f"Error in edge_text_to_speech: {e}")
        return False

def speach_recognition():
    """Speech recognition using Google's speech recognition API"""
    try:
        recognizer = speech_recognition.Recognizer()
        # Improve parameters for better recognition
        recognizer.energy_threshold = 4000  # Adjust based on your microphone
        recognizer.dynamic_energy_threshold = True
        
        with speech_recognition.Microphone() as mic:
            print("Adjusting for ambient noise...")
            # Increase duration for better ambient noise adjustment
            recognizer.adjust_for_ambient_noise(mic, duration=1.0)
            
            print("Listening...")
            # Add timeout and phrase_time_limit for better user experience
            audio = recognizer.listen(mic, timeout=5, phrase_time_limit=10)
            
            print("Recognizing...")
            text = recognizer.recognize_google(audio)
            text = text.lower()
            print(f"Recognized: {text}")
            return text
    except speech_recognition.WaitTimeoutError:
        print("Listening timed out. No speech detected.")
        return ""
    except speech_recognition.UnknownValueError:
        print("Could not understand audio")
        return ""
    except Exception as e:
        print(f"Error in speech recognition: {e}")
        return ""

def AWS_text_to_speech(ai_response):
    """AWS Polly Text-to-Speech"""
    try:
        # Clean the text of markdown symbols
        cleaned_text = clean_text_for_tts(ai_response)
        
        print("Starting text-to-speech")
        polly_client = boto3.client(
            "polly", 
            region_name="us-east-1",
            aws_access_key_id=config.AWS_ACCESS,
            aws_secret_access_key=config.AWS_SECRET
        )
        
        response = polly_client.synthesize_speech(
            Text=cleaned_text,
            VoiceId="Amy",  # British Female Voice
            OutputFormat="mp3",
            Engine="neural"  # Use neural engine for better quality
        )
        
        output_file = "output.mp3"
        if os.path.exists(output_file):
            os.remove(output_file)
            
        with open(output_file, "wb") as file:
            file.write(response["AudioStream"].read())
        
        playsound(output_file, block=True)
        time.sleep(1.2)  # Add some delay to prevent cutting off
        return True
    except Exception as e:
        print(f"Error in AWS_text_to_speech: {e}")
        return False
    



import speech_recognition as sr
import whisper
import tempfile
import os
import queue
import threading
import time

# Audio data queue and stop event for continuous listening
audio_queue = queue.Queue()
stop_listening = threading.Event()
model = whisper.load_model("medium")

def continuous_recorder():
    """Records audio continuously in a background thread"""
    recognizer = sr.Recognizer()
    # Make it more tolerant of pauses
    recognizer.pause_threshold = 2.0  # Default is 0.8 seconds
    recognizer.energy_threshold = 1000
    recognizer.dynamic_energy_threshold = True
    
    with sr.Microphone() as source:
        recognizer.adjust_for_ambient_noise(source, duration=1.0)
        print("Listening continuously...")
        
        while not stop_listening.is_set():
            try:
                # Use a shorter timeout but NO phrase time limit
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=None)
                audio_queue.put(audio)
                # Small delay to prevent CPU overload
                time.sleep(0.1)
            except sr.WaitTimeoutError:
                continue  # Just keep listening
            except Exception as e:
                print(f"Error in recording: {e}")
                break

def whisper_browser_audio_recognition():
    """Use Whisper to transcribe audio recorded from the browser"""
    # Path to the temporary audio file
    temp_audio_file = os.path.join(os.path.dirname(__file__), "temp_audio.wav")
    
    # Wait for the file to exist (with timeout)
    start_time = time.time()
    while not os.path.exists(temp_audio_file) and time.time() - start_time < 30:
        print("Waiting for audio file from browser...")
        time.sleep(1)
    
    if not os.path.exists(temp_audio_file):
        print("No audio file received from browser.")
        return ""
    
    try:
        # Let the file finish writing
        time.sleep(0.5)
        
        # Get file modification time
        mod_time = os.path.getmtime(temp_audio_file)
        
        # Check if file is recently modified (within last 30 seconds)
        if time.time() - mod_time > 30:
            print("Audio file is too old. Please record new audio.")
            return ""
        
        # Transcribe with Whisper
        print("Transcribing browser audio...")
        result = model.transcribe(temp_audio_file, language="fr")
        transcription = result["text"].strip()
        
        print(f"Transcription: {transcription}")
        
        # Delete the file after processing
        os.remove(temp_audio_file)
        
        return transcription
    
    except Exception as e:
        print(f"Error transcribing browser audio: {e}")
        return ""