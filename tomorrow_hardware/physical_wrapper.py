#!/usr/bin/env python3
"""
Physical GPIO Wrapper for Cypher
Adds LEDs and button without modifying main codebase
Just run this instead of main.py
"""

import sys
import os
import time
import asyncio

# Check if on Pi
try:
    import RPi.GPIO as GPIO
    ON_PI = True
except ImportError:
    print("⚠️  Not on Pi - GPIO will be simulated")
    ON_PI = False
    
    # Fake GPIO
    class GPIO:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        HIGH = 1
        LOW = 0
        FALLING = "FALLING"
        PUD_UP = "PUD_UP"
        
        @staticmethod
        def setmode(mode): pass
        @staticmethod
        def setup(pin, mode, pull_up_down=None): pass
        @staticmethod
        def output(pin, state): 
            print(f"[GPIO] Pin {pin} -> {'HIGH' if state else 'LOW'}")
        @staticmethod
        def input(pin): return 0
        @staticmethod
        def add_event_detect(pin, edge, callback=None, bouncetime=200): pass
        @staticmethod
        def cleanup(): pass

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# GPIO Configuration
STATUS_LED = 17   # Green - Running
THINK_LED = 27    # Yellow - Processing  
ERROR_LED = 22    # Red - Error
BUTTON = 23       # Input button

# Setup GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setup(STATUS_LED, GPIO.OUT)
GPIO.setup(THINK_LED, GPIO.OUT)
GPIO.setup(ERROR_LED, GPIO.OUT)
GPIO.setup(BUTTON, GPIO.IN, pull_up_down=GPIO.PUD_UP)

# Status indication
def startup_sequence():
    """LED startup animation"""
    print("🚀 Starting Cypher with GPIO...")
    for _ in range(3):
        GPIO.output(STATUS_LED, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(THINK_LED, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(ERROR_LED, GPIO.HIGH)
        time.sleep(0.1)
        GPIO.output(STATUS_LED, GPIO.LOW)
        GPIO.output(THINK_LED, GPIO.LOW)
        GPIO.output(ERROR_LED, GPIO.LOW)
        time.sleep(0.1)
    GPIO.output(STATUS_LED, GPIO.HIGH)  # Ready

def button_pressed(channel):
    """Handle button press"""
    print("🔘 Button pressed!")
    # Flash think LED
    for _ in range(3):
        GPIO.output(THINK_LED, GPIO.HIGH)
        time.sleep(0.05)
        GPIO.output(THINK_LED, GPIO.LOW)
        time.sleep(0.05)

# Add button detection
GPIO.add_event_detect(BUTTON, GPIO.FALLING, callback=button_pressed, bouncetime=200)

# Now import and wrap main Cypher
try:
    from main import main
    from interfaces.chat import ChatManager
    
    # Monkey-patch ChatManager to add LEDs
    original_discussion = ChatManager.discussion_turn
    
    async def discussion_with_gpio(self):
        """Wrapped discussion with GPIO indicators"""
        GPIO.output(THINK_LED, GPIO.HIGH)  # Start thinking
        
        try:
            result = await original_discussion(self)
            GPIO.output(THINK_LED, GPIO.LOW)  # Done thinking
            
            # Flash status LED for successful response
            for _ in range(2):
                GPIO.output(STATUS_LED, GPIO.LOW)
                await asyncio.sleep(0.1)
                GPIO.output(STATUS_LED, GPIO.HIGH)
                await asyncio.sleep(0.1)
                
            return result
            
        except Exception as e:
            GPIO.output(THINK_LED, GPIO.LOW)
            GPIO.output(ERROR_LED, GPIO.HIGH)
            await asyncio.sleep(1)
            GPIO.output(ERROR_LED, GPIO.LOW)
            raise e
    
    # Replace method
    ChatManager.discussion_turn = discussion_with_gpio
    
    # Also wrap process_conversation_turn for LED feedback
    original_process = ChatManager.process_conversation_turn
    
    async def process_with_gpio(self, user_input):
        """Wrapped processing with GPIO"""
        GPIO.output(THINK_LED, GPIO.HIGH)
        result = await original_process(self, user_input)
        GPIO.output(THINK_LED, GPIO.LOW)
        return result
    
    ChatManager.process_conversation_turn = process_with_gpio
    
    print("✅ Successfully wrapped main Cypher with GPIO")
    
except ImportError as e:
    print(f"⚠️  Could not import main Cypher: {e}")
    print("Falling back to minimal test mode...")
    
    # Fallback minimal mode
    async def main():
        print("Running minimal GPIO test mode")
        GPIO.output(STATUS_LED, GPIO.HIGH)
        
        while True:
            await asyncio.sleep(1)
            GPIO.output(THINK_LED, GPIO.HIGH)
            await asyncio.sleep(0.5)
            GPIO.output(THINK_LED, GPIO.LOW)

# Run with cleanup
if __name__ == "__main__":
    try:
        startup_sequence()
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")
    finally:
        # Cleanup
        GPIO.output(STATUS_LED, GPIO.LOW)
        GPIO.output(THINK_LED, GPIO.LOW)
        GPIO.output(ERROR_LED, GPIO.LOW)
        GPIO.cleanup()
        print("✨ GPIO cleaned up")