##main
from core import memory
from core import nlp
from interfaces import chat
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def run():
    while True:
        user_input = chat.discussion()
        if user_input == "exit":
            print("Bye.................")
            break




run()


