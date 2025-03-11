##chat

import requests
import sys
import os
from core import nlp
from core import memory

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import config

def discussion():
    convos = memory.get_convos()
    prompt = input("you: ")
    past_messages = memory.get_convos()
    convos.extend(past_messages)
    convos.append({"role" : "user" , "parts" : [{"text" : prompt}]})
    data = {"contents" : convos,
            "generationConfig" : {
                "temperature" : 0.72
            }}
    if (prompt == "exit"):
        return "exit"
    ai_response = nlp.send_request(data)
    print(f"{config.MODEL_NAME}: {ai_response}")
    memory.save_convos("user", prompt)
    memory.save_convos("model", ai_response)

    return prompt
