##chat
import requests
import sys
import os
from core import nlp
from core import memory
from interfaces import speech

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

            speech.best_text_to_speech(ai_response)
    except requests.exceptions.RequestException:
        speech.text_to_speech(ai_response)
        return audio_prompt



def text_convos():
    convos = memory.get_convos()
    past_messages = memory.get_convos()
    convos.extend(past_messages)
    text_prompt = input("you: ")
    convos.append({"role" : "user" , "parts" : [{"text" : text_prompt}]})
    data = {"contents" : convos,
            "generationConfig" : {
                "temperature" : 0.72
            }}
    if (text_prompt == "exit"):
        return "exit"
    ai_response = nlp.send_request(data)
    print(f"{config.MODEL_NAME}: {ai_response}")
    memory.save_convos("user", text_prompt)
    memory.save_convos("model", ai_response)
    try:
        check_online = requests.get("https://www.google.com")
        if check_online.status_code == 200:

            speech.best_text_to_speech(ai_response)
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
