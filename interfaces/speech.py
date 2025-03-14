import os
import pyttsx3
from interfaces import chat
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


engine = pyttsx3.init()


def text_to_speech (ai_response):
    engine.setProperty("rate" , 150)
    engine.say(f"{ai_response}")
    engine.runAndWait()

def google_text_to_speech (ai_response):
    language = 'en'
    ai_speech = gTTS(text = ai_response, lang = language, slow = False)
    if os.path.exists("my_speech.mp3"):
        os.remove("my_speech.mp3")
    ai_speech.save("my_speech.mp3")
    playsound("my_speech.mp3")


def elevenLab_text_to_speech(ai_response):

    client = ElevenLabs(api_key = config.VOICE_API_KEY)
    audio_stream = client.text_to_speech.convert_as_stream(
        text= ai_response,
        voice_id="pFZP5JQG7iQjIQuC4Bku",
        model_id="eleven_multilingual_v2"
    )
    # option 1: play the streamed audio locally
    stream(audio_stream)


async def text_to_mp3(ai_response, filename = "response.mp3"):
    communicate = edge_tts.Communicate(ai_response, "en-GB-SoniaNeural")
    if os.path.exists(filename):
        os.remove(filename)
    await communicate.save(filename)  
    return filename

def edge_text_to_speech(ai_response):
    filename= "response.mp3"
    asyncio.run(text_to_mp3(ai_response, filename))
    playsound(filename)



def speach_recognition():
    recognizer = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.012)
        audio = recognizer.listen(mic)

        text = recognizer.recognize_google(audio)
        text = text.lower()
        return text


def AWS_text_to_speech(ai_response):
    print("Starting text-to-speech")
    polly_client = boto3.client("polly", region_name="us-east-1")
    
    response = polly_client.synthesize_speech(
        Text= ai_response,
        VoiceId="Amy",  # British Female Voice
        OutputFormat="mp3"
    )
    if os.path.exists("output.mp3"):
        os.remove("output.mp3")
    with open("output.mp3", "wb") as file:
        file.write(response["AudioStream"].read())
    
    playsound("output.mp3", block=True)
    time.sleep(1.2)
    return True
    
    








    
