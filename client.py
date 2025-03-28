# client.py
import argparse
import requests
import tempfile
import os
import time
import subprocess
import sys

def play_audio(audio_file):
    """Play audio file using appropriate system command"""
    if sys.platform == 'darwin':  # macOS
        subprocess.run(['afplay', audio_file])
    elif sys.platform.startswith('linux'):  # Linux
        subprocess.run(['aplay', audio_file])
    elif sys.platform == 'win32':  # Windows
        import winsound
        winsound.PlaySound(audio_file, winsound.SND_FILENAME)
    else:
        print(f"Unsupported platform: {sys.platform}")
        print(f"Audio saved to: {audio_file}")

def text_to_speech(text, url, voice=None):
    """Send text to TTS API and play the returned audio"""
    print(f"Sending text to TTS API: {text[:50]}...")
    
    data = {"text": text}
    if voice:
        data["voice"] = voice
    
    try:
        # Send request to the API
        response = requests.post(f"{url}/tts", json=data, stream=True)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return
        
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
            temp_filename = f.name
        
        # Play the audio
        play_audio(temp_filename)
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
    except Exception as e:
        print(f"Error: {e}")

def list_voices(url):
    """Get available voices from the API"""
    try:
        response = requests.get(f"{url}/voices")
        if response.status_code == 200:
            voices = response.json().get('voices', [])
            print("Available voices:")
            for voice in voices:
                print(f"- {voice}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description='TTS Client')
    parser.add_argument('--url', default='http://localhost:8080', help='TTS API URL')
    parser.add_argument('--voice', help='Voice to use (if supported by the engine)')
    parser.add_argument('--list-voices', action='store_true', help='List available voices')
    parser.add_argument('text', nargs='?', help='Text to convert to speech')
    
    args = parser.parse_args()
    
    if args.list_voices:
        list_voices(args.url)
        return
    
    if not args.text:
        print("Please provide text to convert to speech")
        return
    
    text_to_speech(args.text, args.url, args.voice)

if __name__ == "__main__":
    main()