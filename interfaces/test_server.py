#!/usr/bin/env python3

import requests
import json
import time
import os
import argparse
import sys

def test_server(server_url):
    """Test the Orpheus TTS server"""
    
    print(f"Testing Orpheus TTS server at {server_url}...")
    
    # Test server health
    try:
        response = requests.get(f"{server_url}/health")
        print(f"Health check: {response.status_code}")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code != 200:
            print("WARNING: Server may not be healthy")
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Could not connect to server: {e}")
        sys.exit(1)
    
    # Test voices endpoint
    try:
        response = requests.get(f"{server_url}/voices")
        print("\nAvailable voices:")
        print(json.dumps(response.json(), indent=2))
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch voices: {e}")
    
    # Test TTS endpoint
    text = "Hello, this is a test of the Orpheus text-to-speech system."
    voice = "tara"  # Default voice
    
    print(f"\nGenerating speech for: '{text}'")
    print(f"Using voice: {voice}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{server_url}/tts",
            json={
                "text": text,
                "voice": voice
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"Success! Audio generated in {result['processing_time']:.2f} seconds.")
            print(f"Audio duration: {result['duration']:.2f} seconds")
            print(f"Audio URL: {server_url}{result['audio_url']}")
            
            # Download the audio file
            audio_url = f"{server_url}{result['audio_url']}"
            audio_response = requests.get(audio_url)
            
            if audio_response.status_code == 200:
                output_file = "test_output.wav"
                with open(output_file, "wb") as f:
                    f.write(audio_response.content)
                print(f"Audio saved to {output_file}")
            else:
                print(f"Failed to download audio: {audio_response.status_code}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"ERROR: Request failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the Orpheus TTS server")
    parser.add_argument("--url", default="http://localhost:8080", help="Server URL")
    args = parser.parse_args()
    
    test_server(args.url)