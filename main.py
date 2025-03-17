##main
from core import memory
from core import nlp
from interfaces import chat
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from interfaces import speech
from services import utilities

def run():
    user_input = False
    while True:
        if not user_input:
            user_input = input("type Delete, to start over, Pass to ignore: ")     
            if user_input.lower() == "delete":
                memory.delete_convos()
                print(f"{config.MODEL_NAME}: Memory Deleted Sucessfully..........")
            user_input = True
        a = chat.discussion()
        if a == "exit":
                print("Bye.................")
                break
        


run()