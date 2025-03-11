###memory
import json
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

convos = []

def save_convos(role, prompt):
    global convos
    convos.append({"role": role, "parts" : [{"text" : prompt}]})
    with open ("past_conversations.json" , "w") as f:
        json.dump(convos, f)



def get_convos():
    if os.path.exists("past_conversations.json"):
        with open ("past_conversations.json" , "r") as f:
            convos = json.load(f)
        return convos 
    else:
        return []
