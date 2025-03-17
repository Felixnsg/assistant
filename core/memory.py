###memory
import json
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from services import utilities


system_prompt = config.SYSTEM_PROMPT

time_prompt = f"{config.time_prompt} {utilities.tell_time()}"

def save_convos(role, prompt):
    global convos
    if not convos:
        convos.append({"role": role, "parts" : [{"text" : system_prompt}]})   
    convos.append({"role": role, "parts" : [{"text" :prompt}]})
    with open ("past_conversations.json" , "w") as f:
        json.dump(convos, f)



def get_convos():
    if os.path.exists("past_conversations.json"):
        with open ("past_conversations.json" , "r") as f:
            convos = json.load(f)
        return convos 
    else:
        return []
convos = get_convos()

def delete_convos():
    if os.path.exists("past_conversations.json"):
        os.remove("past_conversations.json")
        global convos
        convos = []
        return convos


    

