# ffplay_test.py
# Simple test for AllTalk with ffplay

import os
import time
import tempfile
import requests
import subprocess
from urllib.parse import quote

# AllTalk server URL
TTS_SERVER_URL = "http://localhost:7851/api/tts-generate-streaming"

def speak_with_ffplay(text, voice="female_01.wav", language="en"):
    """
    Speak text using AllTalk and ffplay.
    
    Args:
        text: The text to speak
        voice: Voice ID to use
        language: Language code
    
    Returns:
        bool: True if successful, False if there was an error
    """
    print(f"Speaking: '{text}'")
    print(f"Voice: {voice}, Language: {language}")
    
    try:
        # Create URL with query parameters
        encoded_text = quote(text)
        output_file = f"stream_{int(time.time())}.wav"
        url = f"{TTS_SERVER_URL}?text={encoded_text}&voice={voice}&language={language}&output_file={output_file}"
        
        print(f"Requesting audio from server...")
        
        # Download the audio
        response = requests.get(url, stream=True, timeout=20)
        
        if response.status_code != 200:
            print(f"Error: Server returned status code {response.status_code}")
            return False
        
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_path = temp_file.name
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    temp_file.write(chunk)
        
        # Check file size
        file_size = os.path.getsize(temp_path)
        print(f"Downloaded audio file: {temp_path} ({file_size} bytes)")
        
        if file_size < 1000:  # Arbitrary check for valid audio
            print(f"Warning: Audio file is very small ({file_size} bytes)")
        
        # Play with ffplay
        print("Playing audio with ffplay...")
        cmd = ["ffplay", "-autoexit", "-nodisp", temp_path]
        
        ffplay_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Wait for ffplay to complete
        ffplay_process.wait()
        
        # Check return code
        if ffplay_process.returncode == 0:
            print("Audio played successfully!")
            result = True
        else:
            print(f"Error: ffplay returned code {ffplay_process.returncode}")
            result = False
        
        # Clean up
        os.unlink(temp_path)
        print(f"Temporary file removed: {temp_path}")
        
        return result
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    print("=== AllTalk with ffplay Test ===")
    
    # Test 1: Basic test
    print("\nTest 1: Basic test")
    speak_with_ffplay("This is a test of AllTalk with ffplay.")
    
    # Test 2: Different voice
    print("\nTest 2: Different voice")
    speak_with_ffplay(
        "This is a test with a different voice.",
        voice="male_01.wav"
    )
    
    # Test 3: Longer text
    print("\nTest 3: Longer text")
    speak_with_ffplay(
        "This is a longer piece of text to test if ffplay can handle longer audio streams. "
        "The quick brown fox jumps over the lazy dog. "
        "We need to ensure that ffplay works well with AllTalk for all types of content."
    )
    
    print("\nTest complete!")

if __name__ == "__main__":
    main()