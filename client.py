"""
RealtimeTTS Client Script

This script runs on your local computer and sends text to the TTS server.
It receives audio data and plays it back locally.

Run this on your local computer with:
    python tts_client.py
"""

import requests
import json
import tempfile
import os
import time
import argparse
import logging
from playsound import playsound
import pygame

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TTSClient:
    def __init__(self, server_url):
        self.server_url = server_url
        self.session = requests.Session()
        
        # Initialize pygame for audio playback
        pygame.mixer.init()
        
        # Check server health
        try:
            response = self.session.get(f"{self.server_url}/health")
            if response.status_code == 200:
                info = response.json()
                logger.info(f"Connected to TTS server: {info}")
            else:
                logger.warning(f"Server responded with status code: {response.status_code}")
        except requests.RequestException as e:
            logger.error(f"Failed to connect to server: {str(e)}")
    
    def text_to_speech(self, text, language="en", play=True):
        """Convert text to speech using the remote TTS server"""
        try:
            logger.info(f"Sending text to TTS server: '{text[:50]}...'")
            start_time = time.time()
            
            response = self.session.post(
                f"{self.server_url}/tts",
                json={"text": text, "language": language},
                stream=False
            )
            
            if response.status_code != 200:
                logger.error(f"Error from server: {response.status_code}")
                if response.headers.get('Content-Type') == 'application/json':
                    logger.error(f"Error details: {response.json()}")
                return False
            
            # Save and play the audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_file.write(response.content)
                temp_path = temp_file.name
            
            total_time = time.time() - start_time
            logger.info(f"Received audio in {total_time:.2f} seconds")
            
            if play:
                logger.info("Playing audio...")
                pygame.mixer.music.load(temp_path)
                pygame.mixer.music.play()
                
                # Wait for audio to finish
                while pygame.mixer.music.get_busy():
                    pygame.time.Clock().tick(10)
                
                logger.info("Audio playback complete")
            
            os.remove(temp_path)
            return True
            
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"Error during TTS processing: {str(e)}")
            return False
    
    def interactive_mode(self):
        """Run an interactive mode where the user can type text for TTS"""
        print("\n===== RealtimeTTS Client Interactive Mode =====")
        print("Type text and press Enter to convert to speech.")
        print("Type 'exit' or 'quit' to end the session.\n")
        
        while True:
            try:
                text = input("> ")
                if text.lower() in ["exit", "quit"]:
                    break
                
                if text:
                    self.text_to_speech(text)
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error: {str(e)}")
        
        print("\nExiting interactive mode.")

def main():
    parser = argparse.ArgumentParser(description="RealtimeTTS Client")
    parser.add_argument("--server", default="http://localhost:8080", 
                        help="TTS server URL (default: http://localhost:8080)")
    parser.add_argument("--text", help="Text to convert to speech")
    parser.add_argument("--interactive", action="store_true", 
                        help="Run in interactive mode")
    
    args = parser.parse_args()
    
    client = TTSClient(args.server)
    
    if args.text:
        client.text_to_speech(args.text)
    elif args.interactive:
        client.interactive_mode()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()