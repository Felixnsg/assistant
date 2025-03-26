##chat
import requests
import sys
import os
import time
from core import nlp
from core import memory
from interfaces import speech
from interfaces import alltalk
from interfaces import streamaudio
from interfaces import streamaudio  # Import the browser_tts module
from services import utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
import json
from IseeYou import IseeYou


def data_prep(prompt: str, convos: list = None) -> dict:
    """Format data in an understandable format for the used LLM

    Args:
        prompt: The user prompt
        convos: The conversation history list

    Returns:
        dict: The formatted dict ready for the LLM
    """
    try:
        convos_formatted = convos.copy() if convos else []
        # Add the current prompt
        convos_formatted.append({"role": "user", "parts": [{"text": prompt}]})
        
        # Create the data structure
        data = {
            "contents": convos_formatted,
            "generationConfig": {
                "temperature": 0.72
            }
        }
        return data
    except (TypeError, AttributeError, ValueError) as e:
        print("There might be an issue with the Format:", e)
        return {}

class ChatManager:
    def __init__(self, memory_instance, nlp_instance, config_instance, utilities_instances, IseeYou_instance):
        self.memory = memory_instance
        self.nlp = nlp_instance
        self.utilities = utilities_instances
        self.config = config_instance
        self.IseeYou = IseeYou_instance
        self.format = self.choose_format()
        self.prompt = ""
        self.ai_response = ""
        self.data = {}
        self.convos = []
        

    def choose_format(self) -> str:
        """Prompt the user to choose a format between text or audio."""
        format_choice = input("What format do you want: ")
        return format_choice

    def call_request(self):
        """Make the API request to get AI response."""
        try:
            # Load past conversations
            self.convos = self.memory.get_convos()
            # Check for exit command
            if self.prompt.lower() == "exit":
                return "exit"
                
            # Prepare data using standalone function
            self.data = data_prep(self.prompt, self.convos)
            
            # Make API request
            self.ai_response = self.nlp.send_request(self.data)
            return self.ai_response
            
        except (TypeError, ValueError, KeyError) as e:
            print("Misconfigured Data Likely: ", e)
        except (requests.exceptions.RequestException, TimeoutError, ConnectionError) as e:
            print("There might be an issue with the call", e)
        return "Sorry, I encountered an error."

    def save_convos(self) -> None:
        """Save the conversation to memory."""
        self.memory.save_convos("user", self.prompt)
        self.memory.save_convos("model", self.ai_response)

    def talker(self) -> None:
        """Handle text-to-speech output."""
        try:
            check_online = requests.get("https://www.google.com")
            if check_online.status_code == 200:
               speech.text_to_speech(self.ai_response)
            else:
                speech.text_to_speech(self.ai_response)
        except requests.exceptions.RequestException:
            speech.text_to_speech(self.ai_response)

    def process_conversation(self, user_input):
        """Process a conversation turn with the user input."""
        self.prompt = user_input
        
        # Check for exit command
        if self.prompt.lower() == "exit":
            return "exit"
        
        self.call_request()
        
        import asyncio
        try:
            asyncio.create_task(self.IseeYou.run(video_source=0))
        except RuntimeError:
            # If not in an async environment, fallback to run manually
            loop = asyncio.get_event_loop()
            loop.create_task(self.IseeYou.run(video_source=0))
        

        if self.utilities.monitor_sypher(self.ai_response):
            print("Toggled")
            self.utilities.choose_service(self.ai_response)
            
            
        # Display the response
        print(f"{self.config.MODEL_NAME}: {self.ai_response}")
        
        # Save conversation and speak response
        self.save_convos()
        self.talker()
        
        return self.ai_response

    def wait_for_audio_to_finish(self, max_wait_time=60):
        """Wait until all audio has finished playing before continuing.
        
        Args:
            max_wait_time (int): Maximum time to wait in seconds
            
        Returns:
            bool: True if audio finished, False if timed out
        """
        print("Waiting for audio to finish before listening...")
        
        # Use the new wait_until_safe_to_listen function from browser_tts
        result = streamaudio.wait_until_safe_to_listen(timeout=max_wait_time)
        
        if result:
            print("Audio finished, now safe to listen")
            return True
        else:
            print(f"Waited {max_wait_time} seconds but audio is still playing.")
            print("Proceeding anyway to prevent hanging...")
            return False

    def audio_convos(self) -> str:
        """Handle audio conversation format with synchronization."""
        # First, wait until any current audio output has finished
        self.wait_for_audio_to_finish()
        
        # Now it's safe to listen
        print("Listening for your voice...")
        audio_prompt = speech.whisper_speech_recognition()
        
        # If we didn't get any input, try again
        if not audio_prompt or audio_prompt.strip() == "":
            print("No speech detected, trying again...")
            return self.audio_convos()
            
        return self.process_conversation(audio_prompt)

    def text_convos(self) -> str:
        """Handle text conversation format."""
        text_prompt = input("you: ")
        return self.process_conversation(text_prompt)

    def discussion(self):
        """Choose the appropriate conversation format."""
        if self.format.lower() == "text":
            return self.text_convos()
        else:
            return self.audio_convos()
        

    