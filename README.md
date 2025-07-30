# Cypher AI - Multi-Modal Assistant System
## 🚀 Features

### 🤖 Core AI Capabilities
- **Conversational AI** powered by Google's Gemini API
- **Persistent Memory** - Remembers conversation history across sessions
- **Multi-modal Interaction** - Text and voice input/output
- **Context-aware Responses** - Integrates visual information into conversations

### 👁️ Computer Vision System
- **Real-time Face Recognition** using YOLOv8 + custom classifier
- **GPU Acceleration** with CUDA support
- **Live Video Processing** via webcam
- **Person Detection & Tracking** with confidence scoring
- **Visual Context Caching** for seamless AI integration

### 🔊 Audio Processing
- **Text-to-Speech (TTS)** with multiple engines:
  - pyttsx3 (offline)
  - Google TTS
  - ElevenLabs (premium)
  - Edge TTS
  - AWS Polly
  - AllTalk TTS with streaming
- **Speech-to-Text (STT)** options:
  - Google Speech Recognition
  - OpenAI Whisper (local and API)
  - Continuous audio processing

### 🛠️ Utility Services
- **Weather Information** via WeatherAPI
- **Time & Date** queries
- **Mood Setting** via YouTube automation
- **Web Automation** using Selenium
- **Service Triggers** - AI can call functions based on conversation

## 📁 Project Structure

```
IseeYou/
├── IseeYou/                    # Computer Vision System
│   ├── GPUserver.py           # GPU-accelerated detection server
│   ├── IseeYouClass.py        # Video client for webcam processing
│   ├── felix_recognizer.py    # Custom face recognition model
│   ├── person_detector.py     # YOLOv8 person detection
│   └── picturecollector.py    # Image collection utility
├── core/                      # Core System Components
│   ├── memory.py              # Conversation memory management
│   ├── nlp.py                 # LLM API integration
│   ├── cache.py               # Visual context caching
│   └── task_manager.py        # Task coordination
├── interfaces/                # User Interfaces
│   ├── chat.py                # Main conversation manager
│   ├── speech.py              # Audio input/output
│   ├── StreamTTSPlayer.py     # Advanced TTS with streaming
│   └── Whisper.py             # Local Whisper STT server
├── services/                  # Utility Services
│   └── utilities.py           # Weather, time, automation services
├── cache_dumps/               # Visual context data storage
└── logs/                      # System logs
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for vision features)
- Webcam (for face recognition)
- Chrome browser (for web automation)

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd IseeYou
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

For CPU-only setup:
```bash
pip install -r requirements.txt
```

For GPU acceleration (recommended):
```bash
pip install -r requirement_GPU.txt
```

### 4. Configure Environment
Create a `config.py` file with your API keys and settings:

```python
# API Keys
GEMINI_KEY = "your_gemini_api_key"
WEATHER_API_KEY = "your_weatherapi_key"
VOICE_API_KEY = "your_elevenlabs_key"  # Optional
AWS_ACCESS = "your_aws_access_key"     # Optional
AWS_SECRET = "your_aws_secret_key"     # Optional

# TTS/STT Configuration
DEFAULT_TTS_ENGINE = "pyttsx3"  # or "google", "elevenlab", "edge", "aws"
DEFAULT_STT_METHOD = "whisper_api"  # or "google"

# Model Configuration
SYSTEM_PROMPT = "You are a helpful AI assistant with visual capabilities."
MODEL_NAME = "Assistant"
```

### 5. Setup Face Recognition Model
Place your trained Felix classifier model at:
```
/root/models/felix_classifier.pth
```

Or update the path in `GPUserver.py`.

## 🚀 Usage

### Quick Start
```bash
python main.py
```

### Individual Components

#### 1. Start the Vision Server (GPU-accelerated)
```bash
python IseeYou/GPUserver.py
```

#### 2. Run the Whisper STT Server
```bash
python interfaces/Whisper.py
```

#### 3. Start the Main Assistant
```bash
python main.py
```

### Example Interactions

**Basic Conversation:**
```
You: Hello!
Assistant: Hi! How can I help you today?
```

**Face Recognition:**
```
You: track felix
Assistant: Should I start the video feed to track Felix?
You: yes
Assistant: Starting video feed now to look for Felix.
```

**Visual Context:**
```
You: do you see him?
Assistant: Let me check the visual data... Yes, I can see Felix with 85% confidence.
```

**Utility Services:**
```
You: what's the weather like?
Assistant: Let me check the weather for you... [Fetches weather data]

You: what time is it?
Assistant: It is currently Wednesday, 24 July 2024 at 14:35 PM.
```

## ⚙️ Configuration

### Vision System
- **Camera ID:** Modify in `IseeYouClass.py` (default: 0)
- **Detection Confidence:** Adjust in `person_detector.py` (default: 0.7)
- **Model Path:** Update in `GPUserver.py` as needed

### Audio Settings
- **TTS Engine:** Set `DEFAULT_TTS_ENGINE` in config
- **STT Method:** Set `DEFAULT_STT_METHOD` in config
- **Voice Settings:** Configure voice IDs and languages

### Service Integration
- **Weather:** Get free API key from WeatherAPI.com
- **YouTube Automation:** Requires Chrome browser
- **Cloud TTS:** Configure API keys for premium services

## 🔍 Monitoring & Debugging

### Log Files
- `logs/main.log` - Main application logs
- `logs/chat.log` - Conversation logs
- `logs/isee_you_client.log` - Vision system logs
- `cache_dumps/` - Visual context data

### Log Analysis
```bash
python log_analyzer.py "logs/isee_you_client.log"
```

### Visual Context Monitoring
Check `cache_dumps/latest.json` for real-time detection data.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **YOLOv8** for object detection
- **PyTorch** for deep learning framework
- **OpenAI Whisper** for speech recognition
- **Google Gemini** for conversational AI
- **OpenCV** for computer vision

## 🚨 Important Notes

### Privacy & Security
- This system processes video and audio data locally
- Face recognition data is stored locally
- API keys should be kept secure
- Consider privacy implications when deploying

### Performance
- GPU acceleration significantly improves performance
- Adjust detection intervals based on your hardware
- Monitor memory usage with continuous video processing

### Troubleshooting

**Common Issues:**
- **GPU not detected:** Install CUDA and appropriate PyTorch version
- **WebSocket connection failed:** Ensure server is running on correct port
- **Audio issues:** Check microphone permissions and device configuration
- **Model loading errors:** Verify model file path and format

**Getting Help:**
- Check logs for detailed error messages
- Use the log analyzer for connection issues
- Ensure all dependencies are properly installed
