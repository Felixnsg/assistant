#!/usr/bin/env python3
"""
CYPHER HARDWARE "IT'S ALIVE!" TEST
Run this first to verify basic hardware works!
No fancy dependencies - just core Python + RPi.GPIO
"""

import time
import sys
import os

# Check if we're on a Raspberry Pi
try:
    import RPi.GPIO as GPIO
    ON_PI = True
    print("✅ Running on Raspberry Pi")
except ImportError:
    print("⚠️  Not on Raspberry Pi - running in simulation mode")
    ON_PI = False
    
    # Fake GPIO for testing on desktop
    class GPIO:
        BCM = "BCM"
        OUT = "OUT"
        HIGH = 1
        LOW = 0
        
        @staticmethod
        def setmode(mode):
            print(f"[SIM] GPIO mode set to {mode}")
        
        @staticmethod
        def setup(pin, mode):
            print(f"[SIM] Pin {pin} set to {mode}")
        
        @staticmethod
        def output(pin, state):
            print(f"[SIM] Pin {pin} -> {'HIGH' if state else 'LOW'}")
        
        @staticmethod
        def cleanup():
            print("[SIM] GPIO cleanup")

# Simple TTS test
try:
    import pyttsx3
    HAS_TTS = True
    print("✅ TTS available (pyttsx3)")
except ImportError:
    print("⚠️  No TTS - install with: pip3 install pyttsx3")
    HAS_TTS = False

# Simple STT test  
try:
    import speech_recognition as sr
    HAS_STT = True
    print("✅ STT available (speech_recognition)")
except ImportError:
    print("⚠️  No STT - install with: pip3 install speechrecognition")
    HAS_STT = False

print("-" * 50)

class CypherHardwareTest:
    def __init__(self):
        # GPIO pins
        self.LED_STATUS = 17  # Pin 11
        self.LED_THINKING = 27  # Pin 13
        self.LED_ERROR = 22  # Pin 15
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.LED_STATUS, GPIO.OUT)
        GPIO.setup(self.LED_THINKING, GPIO.OUT)
        GPIO.setup(self.LED_ERROR, GPIO.OUT)
        
        # TTS engine
        self.tts = None
        if HAS_TTS:
            try:
                self.tts = pyttsx3.init()
                self.tts.setProperty('rate', 150)
            except:
                print("⚠️  TTS init failed")
                self.tts = None
    
    def led_test(self):
        """Test all LEDs"""
        print("\n🔦 LED TEST")
        print("-" * 30)
        
        leds = [
            (self.LED_STATUS, "Status (Green)"),
            (self.LED_THINKING, "Thinking (Yellow)"),
            (self.LED_ERROR, "Error (Red)")
        ]
        
        for pin, name in leds:
            print(f"Testing {name} LED on pin {pin}...")
            for _ in range(3):
                GPIO.output(pin, GPIO.HIGH)
                time.sleep(0.2)
                GPIO.output(pin, GPIO.LOW)
                time.sleep(0.2)
            print(f"  ✓ {name} LED works!")
        
        return True
    
    def speaker_test(self):
        """Test text-to-speech"""
        print("\n🔊 SPEAKER TEST")
        print("-" * 30)
        
        if not self.tts:
            print("  ✗ No TTS available")
            return False
        
        messages = [
            "Cypher system initializing",
            "Audio output confirmed",
            "Hello, I am alive!"
        ]
        
        for msg in messages:
            print(f"  Speaking: '{msg}'")
            GPIO.output(self.LED_STATUS, GPIO.HIGH)
            self.tts.say(msg)
            self.tts.runAndWait()
            GPIO.output(self.LED_STATUS, GPIO.LOW)
            time.sleep(0.5)
        
        print("  ✓ Speaker works!")
        return True
    
    def microphone_test(self):
        """Test microphone input"""
        print("\n🎤 MICROPHONE TEST")
        print("-" * 30)
        
        if not HAS_STT:
            print("  ✗ No STT available")
            return False
        
        try:
            r = sr.Recognizer()
            with sr.Microphone() as source:
                # Indicate listening
                print("  Adjusting for ambient noise...")
                GPIO.output(self.LED_THINKING, GPIO.HIGH)
                r.adjust_for_ambient_noise(source, duration=1)
                GPIO.output(self.LED_THINKING, GPIO.LOW)
                
                print("  🎤 Say something! (5 seconds)")
                GPIO.output(self.LED_STATUS, GPIO.HIGH)
                
                audio = r.listen(source, timeout=5, phrase_time_limit=5)
                
                GPIO.output(self.LED_STATUS, GPIO.LOW)
                GPIO.output(self.LED_THINKING, GPIO.HIGH)
                
                print("  Processing...")
                text = r.recognize_google(audio)
                
                GPIO.output(self.LED_THINKING, GPIO.LOW)
                
                print(f"  ✓ Heard: '{text}'")
                
                if self.tts:
                    self.tts.say(f"I heard you say: {text}")
                    self.tts.runAndWait()
                
                return True
                
        except sr.WaitTimeoutError:
            print("  ✗ No speech detected")
            GPIO.output(self.LED_ERROR, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(self.LED_ERROR, GPIO.LOW)
            return False
        except sr.UnknownValueError:
            print("  ✗ Could not understand audio")
            return False
        except Exception as e:
            print(f"  ✗ Microphone error: {e}")
            return False
    
    def integration_test(self):
        """Full integration test"""
        print("\n🚀 INTEGRATION TEST")
        print("-" * 30)
        
        # Startup sequence
        print("Starting Cypher sequence...")
        
        # LED dance
        for _ in range(2):
            GPIO.output(self.LED_STATUS, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(self.LED_THINKING, GPIO.HIGH)
            time.sleep(0.1)
            GPIO.output(self.LED_ERROR, GPIO.HIGH)
            time.sleep(0.1)
            
            GPIO.output(self.LED_STATUS, GPIO.LOW)
            GPIO.output(self.LED_THINKING, GPIO.LOW)
            GPIO.output(self.LED_ERROR, GPIO.LOW)
            time.sleep(0.1)
        
        # Announce ready
        if self.tts:
            GPIO.output(self.LED_STATUS, GPIO.HIGH)
            self.tts.say("Cypher is online and ready")
            self.tts.runAndWait()
            GPIO.output(self.LED_STATUS, GPIO.LOW)
        
        print("  ✓ Integration test complete!")
        return True
    
    def cleanup(self):
        """Clean up GPIO"""
        GPIO.cleanup()
        print("\n✅ GPIO cleaned up")

def main():
    print("\n" + "=" * 50)
    print("CYPHER HARDWARE TEST SUITE")
    print("=" * 50)
    
    # System info
    if ON_PI:
        try:
            import subprocess
            temp = subprocess.check_output(['vcgencmd', 'measure_temp']).decode()
            print(f"🌡️  CPU Temperature: {temp.strip()}")
        except:
            pass
    
    # Run tests
    tester = CypherHardwareTest()
    
    results = {
        "LED Test": tester.led_test(),
        "Speaker Test": tester.speaker_test(),
        "Microphone Test": tester.microphone_test(),
        "Integration Test": tester.integration_test()
    }
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST RESULTS")
    print("=" * 50)
    
    for test, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test}: {status}")
    
    # Cleanup
    tester.cleanup()
    
    # Final message
    if all(results.values()):
        print("\n🎉 ALL TESTS PASSED! CYPHER IS ALIVE! 🎉")
        if tester.tts:
            tester.tts.say("All systems operational. Cypher is alive!")
            tester.tts.runAndWait()
    else:
        print("\n⚠️  Some tests failed - check connections and dependencies")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        GPIO.cleanup()
    except Exception as e:
        print(f"\n❌ Critical error: {e}")
        GPIO.cleanup()
        sys.exit(1)