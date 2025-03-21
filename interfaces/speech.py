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

def whisper_speech_recognition():
    """Main function for speech recognition using Whisper with continuous listening"""
    stop_listening.clear()  # Reset stop flag
    audio_queue.queue.clear()  # Clear any previous audio

    # Start recording in background thread
    listen_thread = threading.Thread(target=continuous_recorder)
    listen_thread.daemon = True
    listen_thread.start()
    
    # Allow recording for a moment before processing
    time.sleep(1)
    
    full_transcript = ""
    start_time = time.time()
    last_speech_time = start_time
    active_listening = True
    
    try:
        print("Speak now (system will automatically detect when you're finished)...")
        
        # Process audio until silence threshold exceeded
        while active_listening:
            current_time = time.time()
            
            # Stop conditions:
            # 1. If silence for more than 2.5 seconds after speech was detected
            # 2. If total recording exceeds 30 seconds
            # 3. If the stop event is set
            if (full_transcript and current_time - last_speech_time > 2.5) or \
               (current_time - start_time > 30) or \
               stop_listening.is_set():
                active_listening = False
                continue
                
            # Process available audio segments
            try:
                if not audio_queue.empty():
                    audio = audio_queue.get(block=False)
                    
                    # Save to temp file
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                    temp_filename = temp_file.name
                    temp_file.close()
                    
                    with open(temp_filename, 'wb') as f:
                        f.write(audio.get_wav_data())
                    
                    # Transcribe with Whisper
                    result = model.transcribe(temp_filename, language="fr")
                    segment_text = result["text"].strip()
                    
                    # Clean up
                    try:
                        os.remove(temp_filename)
                    except:
                        pass
                    
                    # If we got text, update the transcript and last speech time
                    if segment_text:
                        print(f"Recognizing: {segment_text}")
                        full_transcript += " " + segment_text
                        last_speech_time = current_time
                else:
                    # No audio available yet, small wait
                    time.sleep(0.1)
                    
            except Exception as e:
                print(f"Error processing audio: {e}")
                
    finally:
        # Clean up
        stop_listening.set()
        listen_thread.join(timeout=1)
        print("Finished listening.")
    
    return full_transcript.strip()