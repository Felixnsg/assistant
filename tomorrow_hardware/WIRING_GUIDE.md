# 🔌 CYPHER WIRING - Make It Alive TODAY!

## Philosophy: "Breadboard First, Pretty Later"

### What We're Building Today:
A working Cypher that can:
1. **Hear** you (USB mic)
2. **Speak** to you (3.5mm speakers)  
3. **Show** it's thinking (LEDs)
4. **Run** your Python code (Raspberry Pi)

---

## 🎯 MINIMAL VIABLE WIRING (Start Here!)

### Shopping List for TODAY:
```
MUST HAVE:
□ Raspberry Pi (any version 3+ works)
□ MicroSD card (8GB+ with Raspbian)
□ USB-C or Micro-USB power (2.5A minimum)
□ ANY USB microphone (even a headset)
□ ANY speakers (3.5mm jack, computer speakers work)
□ 3 LEDs + resistors (or WS2812 strip)
□ Breadboard + jumper wires

NICE TO HAVE:
□ USB sound card (better audio)
□ Level shifter (for 5V LEDs)
□ Multimeter (debugging)
```

---

## 📐 ACTUAL WIRING DIAGRAM

### Version 1: ULTRA SIMPLE (Just Make It Talk!)
```
RASPBERRY PI CONNECTIONS:
═══════════════════════════════════════════════════

USB PORTS:
[USB1] → USB Microphone (any USB mic)
[USB2] → (keep free for keyboard)
[USB3] → (keep free for mouse)  
[USB4] → (keep free)

AUDIO:
[3.5mm Jack] → Speakers/Headphones

POWER:
[USB-C/PWR] → 5V Power Supply (2.5A+)

STATUS LED (Super Simple):
Pin 11 (GPIO17) →→→ [330Ω resistor] →→→ LED(+) 
Pin 9  (GND)    →→→ LED(-)

That's it! This will work!
```

### Version 2: PROPER SETUP (After V1 Works)
```
FULL GPIO PINOUT:
        3.3V (1)  (2) 5V
   SDA/GPIO2 (3)  (4) 5V
   SCL/GPIO3 (5)  (6) GND
       GPIO4 (7)  (8) GPIO14/TXD
         GND (9)  (10) GPIO15/RXD
★    GPIO17 (11)  (12) GPIO18      ★ PWM
      GPIO27 (13)  (14) GND
      GPIO22 (15)  (16) GPIO23
        3.3V (17)  (18) GPIO24
★ MOSI/GPIO10(19)  (20) GND
★ MISO/GPIO9 (21)  (22) GPIO25
★ SCLK/GPIO11(23)  (24) GPIO8/CE0  ★ SPI
         GND (25)  (26) GPIO7/CE1

CONNECTIONS:
Pin 11 → Status LED (through 330Ω)
Pin 12 → WS2812 Data In (needs level shifter!)
Pin 2  → WS2812 5V Power
Pin 6  → Common Ground
```

---

## 🚀 BOOTSTRAP SEQUENCE

### Step 1: Basic Pi Setup (30 mins)
```bash
# 1. Flash Raspbian to SD card
# 2. Boot Pi with monitor/keyboard
# 3. Enable SSH and configure WiFi:
sudo raspi-config

# 4. Update system:
sudo apt update && sudo apt upgrade -y

# 5. Install audio tools:
sudo apt install -y alsa-utils pulseaudio sox

# 6. Test speakers:
speaker-test -t wav -c 2

# 7. Test microphone:
arecord -l  # List recording devices
arecord -d 5 test.wav  # Record 5 seconds
aplay test.wav  # Play it back
```

### Step 2: Install Cypher Dependencies (20 mins)
```bash
# Core Python packages
sudo apt install -y python3-pip python3-venv git

# Audio libraries
sudo apt install -y portaudio19-dev python3-pyaudio

# GPIO for LEDs
sudo apt install -y python3-rpi.gpio

# Create virtual environment
python3 -m venv cypher_env
source cypher_env/bin/activate

# Install Cypher requirements
pip install speechrecognition
pip install pyttsx3
pip install google-generativeai  # For Gemini
```

### Step 3: The "Hello World" Test (5 mins)
```python
# save as: test_alive.py
import RPi.GPIO as GPIO
import time
import pyttsx3
import speech_recognition as sr

# LED setup
LED_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)

# TTS setup
engine = pyttsx3.init()

# Blink LED and speak
for i in range(3):
    GPIO.output(LED_PIN, GPIO.HIGH)
    print(f"Blink {i+1}")
    time.sleep(0.5)
    GPIO.output(LED_PIN, GPIO.LOW)
    time.sleep(0.5)

engine.say("Hello! Cypher is alive!")
engine.runAndWait()

# Listen test
r = sr.Recognizer()
with sr.Microphone() as source:
    print("Say something!")
    audio = r.listen(source, timeout=5)
    try:
        text = r.recognize_google(audio)
        print(f"You said: {text}")
        engine.say(f"You said: {text}")
        engine.runAndWait()
    except:
        print("Couldn't understand")

GPIO.cleanup()
print("✅ CYPHER IS ALIVE!")
```

---

## 🔧 COMPONENT TEST SCRIPTS

### Test 1: LED Heartbeat
```python
# heartbeat.py - Know it's running
import RPi.GPIO as GPIO
import time

GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

# PWM for fade effect
pwm = GPIO.PWM(17, 100)
pwm.start(0)

try:
    while True:
        # Fade in
        for dc in range(0, 101, 5):
            pwm.ChangeDutyCycle(dc)
            time.sleep(0.02)
        # Fade out
        for dc in range(100, -1, -5):
            pwm.ChangeDutyCycle(dc)
            time.sleep(0.02)
except KeyboardInterrupt:
    pass

pwm.stop()
GPIO.cleanup()
```

### Test 2: Audio I/O Check
```python
# audio_test.py - Verify mic and speakers
import sounddevice as sd
import numpy as np

print("Recording 3 seconds...")
recording = sd.rec(int(3 * 44100), samplerate=44100, channels=1)
sd.wait()
print("Playing back...")
sd.play(recording, 44100)
sd.wait()
print("Done!")
```

### Test 3: Full System Check
```python
# system_check.py - Verify everything works
import subprocess
import psutil

def check_system():
    checks = {
        "CPU Temp": lambda: f"{psutil.sensors_temperatures()['cpu_thermal'][0].current}°C",
        "RAM Free": lambda: f"{psutil.virtual_memory().available / 1024**2:.0f}MB",
        "Disk Free": lambda: f"{psutil.disk_usage('/').free / 1024**3:.1f}GB",
        "Audio Out": lambda: "✓" if "bcm2835" in subprocess.getoutput("aplay -l") else "✗",
        "Audio In": lambda: "✓" if "USB" in subprocess.getoutput("arecord -l") else "✗",
        "Python": lambda: subprocess.getoutput("python3 --version"),
    }
    
    for name, check in checks.items():
        try:
            result = check()
            print(f"{name}: {result}")
        except:
            print(f"{name}: ERROR")

check_system()
```

---

## 🎮 RUNNING CYPHER ON PI

### Quick Test:
```bash
# Copy your code to Pi
scp -r /path/to/cypherv001 pi@raspberrypi.local:~/

# SSH into Pi
ssh pi@raspberrypi.local

# Run Cypher
cd cypherv001
python3 main.py
```

### Auto-Start on Boot:
```bash
# Create service file
sudo nano /etc/systemd/system/cypher.service

# Add this content:
[Unit]
Description=Cypher AI Assistant
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/cypherv001
ExecStart=/usr/bin/python3 /home/pi/cypherv001/main.py
Restart=always

[Install]
WantedBy=multi-user.target

# Enable service
sudo systemctl enable cypher
sudo systemctl start cypher
```

---

## 🔴 TROUBLESHOOTING

**"No audio devices found"**
```bash
# Check audio devices
aplay -l  # List playback
arecord -l  # List recording

# Set default device
sudo nano /etc/asound.conf
# Add:
pcm.!default {
    type hw
    card 1  # Your USB audio device
}
```

**"GPIO already in use"**
```bash
# Kill any Python processes
pkill -f python

# Or add to your code:
GPIO.setwarnings(False)
```

**"Import error: No module X"**
```bash
# Always use pip3, not pip
pip3 install missing_module

# Or for system-wide:
sudo apt install python3-missing-module
```

---

## 🏁 SUCCESS CHECKLIST

By end of tomorrow, you should have:

### Hardware Connected:
- [ ] Pi boots successfully
- [ ] USB mic recognized (`arecord -l` shows it)
- [ ] Speakers make sound (`speaker-test` works)
- [ ] LED blinks with GPIO control
- [ ] Can SSH into Pi from laptop

### Software Working:
- [ ] Python 3 installed
- [ ] Can record audio via mic
- [ ] Can play audio via speakers
- [ ] LED responds to code
- [ ] Basic STT/TTS works

### Cypher Running:
- [ ] main.py starts without errors
- [ ] Says "Hello" on startup
- [ ] Responds to voice commands
- [ ] LED indicates status
- [ ] Runs for 5+ minutes stable

---

## 💡 Tomorrow's Plan:

**Hour 1**: Wire everything on breadboard
**Hour 2**: Install Raspbian, test audio
**Hour 3**: Install dependencies, test scripts
**Hour 4**: Run Cypher, debug issues

Remember: We're not building a product yet, we're proving it WORKS. Breadboard and jumper wires are perfect. Make it pretty later!

The goal: When you say "Hey Cypher", an LED lights up and it responds. That's the magic moment! 🎉