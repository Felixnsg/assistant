# 🔧 Cypher Final Component Selection
## Production-Grade Hardware for AI Assistant

### 🧠 COMPUTE MODULE
**Primary Options (Choose based on budget/availability):**

#### Option A: NVIDIA Jetson Orin Nano ($499)
```
- 8GB RAM, 6-core ARM CPU
- 1024-core GPU (40 TOPS)
- Native AI acceleration
- Direct MIPI CSI camera support
- Real-time vision processing
- Power: 15W max
```

#### Option B: Raspberry Pi 5 8GB + Hailo-8 ($150 + $200)
```
- Pi 5: Quad-core 2.4GHz, 8GB RAM
- Hailo-8: 26 TOPS AI accelerator (M.2)
- More modular/repairable
- Better community support
- Power: 12W combined
```

#### Option C: Intel NUC with N100 ($250)
```
- x86 architecture (better software compatibility)
- 4-core, up to 16GB RAM
- Intel UHD Graphics (basic AI)
- More storage options
- Power: 15W TDP
```

---

### 📷 VISION SYSTEM

#### Primary Camera: OAK-D Lite ($149)
```
Part: Luxonis OAK-D-Lite
Why: Depth perception + onboard AI
- Stereo depth sensing
- 4K color camera
- Onboard Intel Myriad X VPU
- Direct neural network inference
- 13 FPS object detection onboard
- USB 3.0 connection
```

#### Alternative: Arducam IMX519 ($60)
```
Part: B0371 Arducam 16MP Autofocus
Why: High quality, autofocus
- 16MP resolution
- Motorized focus
- MIPI CSI-2 direct to Pi/Jetson
- Wide angle options available
- Low light performance
```

---

### 🎤 AUDIO INPUT

#### Microphone Array: ReSpeaker Mic Array v2.0 ($79)
```
Part: Seeed Studio ReSpeaker Mic Array v2.0
Why: Professional voice capture
- 4 MEMS microphones
- Hardware AEC (echo cancellation)
- Hardware noise suppression
- Beamforming capability
- 12 RGB LEDs for status
- USB connection
- Voice Activity Detection onboard
```

#### Alternative: Matrix Voice ESP32 ($65)
```
Part: MATRIX Voice ESP32
Why: More hackable
- 8 MEMS microphones
- ESP32 onboard (WiFi/BT)
- FPGA for audio processing
- 18 RGBW LEDs
- GPIO expansion
```

---

### 🔊 AUDIO OUTPUT

#### Speakers: Dayton Audio RS100-4 ($35 each)
```
Part: RS100-4 4" Reference Full-Range
Why: Audiophile quality in small size
- 4" full-range driver
- 86.4 dB sensitivity
- 20W RMS power handling
- Clear voice reproduction
- No crossover needed
```

#### Amplifier: Adafruit MAX98357 I2S ($7)
```
Part: Adafruit 3006
Why: Digital audio, no DAC needed
- I2S digital input
- 3W output
- Direct GPIO connection
- No analog conversion loss
- Tiny footprint
```

---

### 🎨 DISPLAY & INTERACTION

#### Primary Display: Waveshare 7.9" DSI ($90)
```
Part: Waveshare 7.9inch DSI LCD
Why: Crisp, responsive touch
- 400×1280 resolution (bar shape!)
- IPS panel, 300cd/m²
- Capacitive touch
- DSI connection (no HDMI needed)
- Perfect for status/info bar
```

#### Status Display: 2.42" OLED ($25)
```
Part: Waveshare 2.42inch OLED
Why: Always-on status
- 128×64 pixels
- SPI/I2C connection
- Ultra low power
- High contrast
- Shows "eyes" or status icons
```

---

### 💡 LIGHTING & FEEDBACK

#### Primary: BTF-LIGHTING WS2812B ECO ($20)
```
Part: BTF-LIGHTING WS2812B ECO 5m
Why: Proven reliability
- 60 LEDs/meter
- Individual addressable
- 5V operation
- Built-in controller
- Cut to length
```

#### Control: Custom PCB or PicoPixel ($12)
```
Part: Adafruit PicoPixel Driver
Why: Level shifting + power management
- Proper 3.3V → 5V conversion
- Handles 8A LED current
- Thermal protection
- Tiny footprint
```

---

### 🎛️ SENSORS SUITE

#### Environment: BME680 ($20)
```
Part: Adafruit BME680
Why: Complete environmental sensing
- Temperature, humidity, pressure
- VOC gas sensing (air quality)
- I2C interface
- 3.3V operation
```

#### Presence: LD2410 mmWave ($8)
```
Part: HLK-LD2410 24GHz Radar
Why: Detects presence without camera
- Detects breathing/micro-movements
- Works through plastic
- 5m range
- Serial/GPIO output
- Privacy-preserving
```

#### Proximity: VL53L5CX ToF ($25)
```
Part: STM VL53L5CX
Why: 8×8 depth array
- 64-zone ranging
- Gesture recognition capable
- 400cm range
- I2C interface
```

#### IMU: ICM-42688-P ($15)
```
Part: TDK InvenSense ICM-42688-P
Why: Know device orientation
- 6-axis (accel + gyro)
- ±16g, ±2000dps range
- Low noise
- Motion detection interrupt
```

---

### ⚡ POWER SYSTEM

#### Main PSU: Mean Well RPS-65-5 ($30)
```
Part: RPS-65-5
Why: Medical grade reliability
- 5V 13A output (65W)
- 88% efficiency
- Over-current/voltage protection
- Fanless operation
- UL/CE certified
```

#### Battery Backup: PiSugar 3 Plus ($80)
```
Part: PiSugar 3 Plus 5000mAh
Why: Seamless UPS functionality
- 5000mAh battery
- Auto switching
- I2C battery monitoring
- RTC included
- Magnetic mounting
```

---

## 🔌 CUSTOM PCB OPPORTUNITIES

### Cypher Control Board v1.0
```
Purpose: Consolidate all connections
Features:
- GPIO breakout with level shifting
- I2S audio routing
- Power distribution (5V, 3.3V, 12V)
- USB hub integration
- Sensor connectors (JST)
- LED driver circuit
- Fan PWM control
- Status LEDs
- Emergency stop circuit

Components:
- STM32F0 for GPIO expansion
- TXS0108E level shifters
- TPS54331 buck converters
- USB2514B hub chip
- JST-SH connectors throughout
```

### Benefits of Custom PCB:
1. **Reliability**: No loose jumper wires
2. **Compact**: Stack above Pi/Jetson
3. **Professional**: Looks production-ready
4. **Expandable**: Add sensors easily
5. **Protected**: Proper ESD protection
6. **Debuggable**: Test points + LEDs

---

## 📅 WEEK-LONG BUILD PLAN

### Day 1 (Monday) - Foundation
- Morning: Basic Pi setup, LED blink
- Afternoon: Audio I/O working

### Day 2 (Tuesday) - Senses
- Morning: Camera + face detection
- Afternoon: Microphone array setup

### Day 3 (Wednesday) - Intelligence
- Morning: Get full Cypher running
- Afternoon: Integrate sensors

### Day 4 (Thursday) - Custom PCB
- Morning: Design PCB in KiCad
- Afternoon: Mill/etch prototype

### Day 5 (Friday) - Integration
- Morning: Assemble everything
- Afternoon: Final demo prep

---

## 💰 BUDGET VERSIONS

### Tier 1: Minimum Viable ($200)
- Raspberry Pi 4 (4GB): $55
- USB Webcam: $25
- USB Mic: $20
- Speakers: $30
- LEDs + misc: $20
- Power supply: $20
- Sensors: $30

### Tier 2: Recommended ($500)
- Raspberry Pi 5 (8GB): $80
- OAK-D Lite: $149
- ReSpeaker Array: $79
- Good speakers + amp: $80
- Display: $90
- Sensors suite: $50
- Power system: $70

### Tier 3: Professional ($1000)
- Jetson Orin Nano: $499
- OAK-D Pro: $299
- Matrix Voice: $65
- Studio monitors: $150
- All sensors: $100
- Custom PCB: $50
- Battery backup: $80

---

## 🛒 WHERE TO SOURCE

**Immediate (likely in maker space):**
- Raspberry Pi
- Basic USB camera
- LEDs (WS2812B)
- Resistors, capacitors
- Breadboards, jumpers

**Order for Tuesday delivery:**
- Adafruit.com (US): Most sensors, displays
- DigiKey.com: Specific chips, connectors
- Amazon: Speakers, power supplies

**Special orders (3-5 days):**
- Luxonis.com: OAK-D cameras
- Seeedstudio.com: ReSpeaker, Grove sensors

---

## 🎯 CRITICAL PATH ITEMS

**Must work by Wednesday:**
1. Compute module (Pi/Jetson)
2. Any USB camera
3. Any USB microphone
4. Speakers + amplification
5. LEDs for status

**Nice to have by Friday:**
1. OAK-D for depth sensing
2. Microphone array for direction
3. OLED for status display
4. mmWave for presence
5. Custom PCB prototype

---

## 📝 NOTES FOR CUSTOM PCB DESIGN

When you get to the PCB stage, consider:

```
Stackable Design:
- 40-pin GPIO header pass-through
- M2.5 mounting holes matching Pi
- Height clearance for heatsinks

Connectors:
- JST-SH 1.0mm for sensors (tiny, locking)
- JST-PH 2.0mm for power (robust)
- USB-C for main power input
- Screw terminals for speakers

Protection:
- TVS diodes on all external I/O
- Polyfuse on USB power
- Flyback diodes on motor/fan outputs
- ESD protection on touch pins

Test Points:
- Power rails (5V, 3.3V, GND)
- I2C bus (SDA, SCL)
- SPI bus (MOSI, MISO, CLK)
- Key GPIO signals

Extras:
- Power LED (green)
- Activity LED (blue)
- Error LED (red)
- Reset button
- Boot mode switch
```

---

Remember: Start simple (breadboard + jumpers), but buy components that will work in the final build. No point buying a $5 mic if you'll replace it with a $79 array on Wednesday!