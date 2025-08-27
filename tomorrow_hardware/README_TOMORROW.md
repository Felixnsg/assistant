# 🔧 Tomorrow's Hardware Plan - Quick Reference

## Files in This Folder:

1. **hardware_alive.py** - Test each component individually
2. **minimal_cypher.py** - Basic voice + LED loop (no dependencies)
3. **camera_test.py** - Face detection with LED feedback
4. **physical_wrapper.py** - Runs main Cypher with GPIO (no code changes!)
5. **WIRING_GUIDE.md** - Complete wiring diagrams

## Tomorrow's Order:

### Hour 1: Basic Setup
```bash
# 1. Wire 3 LEDs to pins 17, 27, 22
# 2. Connect USB mic and speakers
# 3. Run:
python3 hardware_alive.py
```

### Hour 2: Voice + LEDs
```bash
# Once hardware test passes:
python3 minimal_cypher.py
# Say "hello" - LED should blink
```

### Hour 3: Try Main Cypher
```bash
cd ..  # Go to main cypherv001
python3 main.py
# See what works/breaks
```

### Hour 4: Main Cypher with GPIO
```bash
cd tomorrow_hardware
python3 physical_wrapper.py
# This runs main.py WITH LED feedback
# No changes to original code!
```

## Minimal Wiring:

```
Pi Pin 11 (GPIO17) → 330Ω → Green LED → GND
Pi Pin 13 (GPIO27) → 330Ω → Yellow LED → GND  
Pi Pin 15 (GPIO22) → 330Ω → Red LED → GND
Pi Pin 16 (GPIO23) → Button → GND (optional)

USB → Microphone
3.5mm → Speakers
```

## What Each LED Means:

- 🟢 Green (Pin 17): System ready/idle
- 🟡 Yellow (Pin 27): Thinking/processing
- 🔴 Red (Pin 22): Error occurred

## If Things Don't Work:

```bash
# Test GPIO access:
gpio readall

# Test audio:
speaker-test -t wav
arecord -l

# Missing modules:
pip3 install RPi.GPIO speechrecognition pyttsx3
```

That's it! Focus on making LEDs respond to code. Everything else is bonus.