#!/usr/bin/env python3
"""
MINIMAL CYPHER - Just Make It Work!
No complex dependencies, just basic Python
This WILL work on your Pi tomorrow!
"""

import os
import sys
import time
import subprocess

print("""
╔══════════════════════════════════════╗
║     CYPHER MINIMAL - LET'S GO!      ║
╚══════════════════════════════════════╝
""")

# Check what we have available
HAS_GPIO = False
HAS_TTS = False
HAS_STT = False

# Try GPIO
try:
    import RPi.GPIO as GPIO
    HAS_GPIO = True
    LED_PIN = 17
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED_PIN, GPIO.OUT)
    print("✅ GPIO ready - LED on pin 17")
except:
    print("⚠️  No GPIO - running without LEDs")

# Try TTS (multiple fallbacks)
try:
    import pyttsx3
    tts_engine = pyttsx3.init()
    HAS_TTS = "pyttsx3"
    print("✅ TTS ready (pyttsx3)")
except:
    # Fallback to espeak
    if subprocess.call("which espeak", shell=True, stdout=subprocess.DEVNULL) == 0:
        HAS_TTS = "espeak"
        print("✅ TTS ready (espeak)")
    else:
        print("⚠️  No TTS - install pyttsx3 or espeak")

# Try STT
try:
    import speech_recognition as sr
    recognizer = sr.Recognizer()
    HAS_STT = True
    print("✅ STT ready (speech_recognition)")
except:
    print("⚠️  No STT - install speechrecognition")

print("-" * 40)

def speak(text):
    """Speak using whatever TTS we have"""
    print(f"🤖 Cypher: {text}")
    
    if HAS_GPIO:
        GPIO.output(LED_PIN, GPIO.HIGH)
    
    if HAS_TTS == "pyttsx3":
        tts_engine.say(text)
        tts_engine.runAndWait()
    elif HAS_TTS == "espeak":
        subprocess.call(f'espeak "{text}"', shell=True)
    
    if HAS_GPIO:
        GPIO.output(LED_PIN, GPIO.LOW)

def listen():
    """Listen using whatever STT we have"""
    if not HAS_STT:
        # Fallback to text input
        return input("You: ")
    
    try:
        with sr.Microphone() as source:
            print("🎤 Listening...")
            if HAS_GPIO:
                # Blink to show listening
                for _ in range(3):
                    GPIO.output(LED_PIN, GPIO.HIGH)
                    time.sleep(0.1)
                    GPIO.output(LED_PIN, GPIO.LOW)
                    time.sleep(0.1)
            
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            if HAS_GPIO:
                GPIO.output(LED_PIN, GPIO.HIGH)  # Processing
            
            text = recognizer.recognize_google(audio)
            print(f"👤 You: {text}")
            
            if HAS_GPIO:
                GPIO.output(LED_PIN, GPIO.LOW)
            
            return text
    except Exception as e:
        print(f"Couldn't hear that: {e}")
        return ""

def process_command(command):
    """Simple command processing"""
    command = command.lower()
    
    if "hello" in command or "hi" in command:
        return "Hello! I'm Cypher, and I'm alive!"
    elif "time" in command:
        current_time = time.strftime("%H:%M")
        return f"The time is {current_time}"
    elif "date" in command:
        current_date = time.strftime("%B %d, %Y")
        return f"Today is {current_date}"
    elif "status" in command:
        return "All systems operational!"
    elif "blink" in command:
        if HAS_GPIO:
            for _ in range(5):
                GPIO.output(LED_PIN, GPIO.HIGH)
                time.sleep(0.2)
                GPIO.output(LED_PIN, GPIO.LOW)
                time.sleep(0.2)
            return "LED test complete!"
        else:
            return "No GPIO available for LED"
    elif "test" in command:
        return "Audio test: one, two, three. Can you hear me?"
    elif "exit" in command or "quit" in command:
        return "QUIT"
    else:
        return f"I heard '{command}'. I'm still learning!"

def main():
    # Startup
    speak("Cypher minimal system starting up")
    time.sleep(0.5)
    speak("Hello! Say 'hello', 'time', 'test', or 'exit'")
    
    # Main loop
    while True:
        try:
            # Get input
            user_input = listen()
            
            if not user_input:
                continue
            
            # Process
            response = process_command(user_input)
            
            if response == "QUIT":
                speak("Goodbye!")
                break
            
            # Respond
            speak(response)
            
        except KeyboardInterrupt:
            print("\nInterrupted!")
            break
        except Exception as e:
            print(f"Error: {e}")
            if HAS_GPIO:
                # Error blink
                for _ in range(10):
                    GPIO.output(LED_PIN, GPIO.HIGH)
                    time.sleep(0.05)
                    GPIO.output(LED_PIN, GPIO.LOW)
                    time.sleep(0.05)
    
    # Cleanup
    if HAS_GPIO:
        GPIO.cleanup()
    print("\n✨ Cypher minimal shutdown complete")

if __name__ == "__main__":
    main()