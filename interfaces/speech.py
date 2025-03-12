import os
import pyttsx3
from interfaces import chat
from gtts import gTTS
from playsound import playsound
import speech_recognition

engine = pyttsx3.init()


def text_to_speech (ai_response):
    engine.setProperty("rate" , 150)
    engine.say(f"{ai_response}")
    engine.runAndWait()

def best_text_to_speech (ai_response):
    language = 'en'
    ai_speech = gTTS(text = ai_response, lang = language, slow = False)
    if os.path.exists("my_speech.mp3"):
        os.remove("my_speech.mp3")
    ai_speech.save("my_speech.mp3")
    playsound("my_speech.mp3")

def speach_recognition():
    recognizer = speech_recognition.Recognizer()
    with speech_recognition.Microphone() as mic:
        recognizer.adjust_for_ambient_noise(mic, duration=0.012)
        audio = recognizer.listen(mic)

        text = recognizer.recognize_google(audio)
        text = text.lower()
        return text

    
    
    







    
