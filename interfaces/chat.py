##chat
import requests
import sys
import os
from core import nlp
from core import memory
from interfaces import speech
from interfaces import alltalk
from interfaces import streamaudio
from services import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

has_run = False
import config




def choose_format():
    format= input("What format do you want: ")
    return format



def audio_convos():
    convos = memory.get_convos()
    past_messages = memory.get_convos()
    convos.extend(past_messages)
    audio_prompt = speech.speach_recognition()
    print(f"you: {audio_prompt}")
    convos.append({"role" : "user" , "parts" : [{"text" : audio_prompt}]})
    data = {"contents" : convos,
            "generationConfig" : {
                "temperature" : 0.72
            }}
    if (audio_prompt == "exit"):
        return "exit"
    ai_response = nlp.send_request(data)
    print(f"{config.MODEL_NAME}: {ai_response}")
    memory.save_convos("user", audio_prompt)
    memory.save_convos("model", ai_response)
    try:
        check_online = requests.get("https://www.google.com")
        if check_online.status_code == 200:
            pass
            streamaudio.say(ai_response)
    except requests.exceptions.RequestException:
        speech.text_to_speech(ai_response)
        return audio_prompt



def text_convos():
    convos = memory.get_convos()
    text_prompt = input("you: ")
    
    if (text_prompt == "exit"):
        return "exit"
        
    convos.append({"role": "user", "parts": [{"text": text_prompt}]})
    data = {"contents": convos,
            "generationConfig": {
                "temperature": 0.72
            }}
    
    ai_response = nlp.send_request(data)
    
    # Check if weather trigger phrases are present
    if utilities.monitor_sypher(ai_response):
        utilities.choose_service(ai_response)
        
    
    
    print(f"{config.MODEL_NAME}: {ai_response}")
    
    # Save conversation history3
    memory.save_convos("user", text_prompt)
    memory.save_convos("model", ai_response)





    try:
        check_online = requests.get("https://www.google.com")
        if check_online.status_code == 200:

            streamaudio.say(ai_response)
            pass
    except requests.exceptions.RequestException:
        speech.text_to_speech(ai_response)
        return text_prompt






global format
format = choose_format()

def discussion():

    if format.lower() == "text":
        return text_convos()
    else:
        return audio_convos()
