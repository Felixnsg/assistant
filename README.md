# Cypher

Multi-modal AI assistant combining conversational AI, live webcam vision, and text/voice I/O in an async Python architecture. A WebSocket-based orchestrator coordinates separate NLP, vision, and audio services.

## What it does

- Conversational chat via Google Gemini with persistent memory across sessions
- Live face detection and recognition over webcam (YOLOv8 for detection, a custom classifier for recognition)
- Text-to-speech across six engines: pyttsx3, Google TTS, ElevenLabs, Edge TTS, AWS Polly, Orpheus
- Speech-to-text via Whisper (local server or API) or Google Speech Recognition
- Utility actions the assistant can trigger: weather lookup, time queries, browser automation via Selenium

## Architecture

An async pipeline orchestrator routes each request through stages: input, context, LLM, response, services, memory, output. Vision runs on a separate GPU server connected over WebSocket. Memory and visual context are cached to disk.

## Layout

- `IseeYou/` - vision pipeline: GPU server, webcam client, YOLOv8 person detection, custom face classifier
- `core/` - memory, NLP wrapper, visual context cache, task coordination
- `interfaces/` - main chat loop, speech I/O, streaming TTS player, Whisper STT server
- `services/` - utility integrations (weather, time, web automation)
- `cache_dumps/` - visual context persistence
- `logs/` - application logs

## Stack

Python (asyncio, WebSockets), PyTorch, YOLOv8, OpenCV, Google Gemini, OpenAI Whisper, Selenium.

## Running it

Requires Python 3.8+, a CUDA-compatible GPU for the vision pipeline, a webcam, and Chrome for browser automation.

```
pip install -r requirements.txt         # CPU
pip install -r requirement_GPU.txt      # GPU
```

Create `config.py` with API keys (Gemini, WeatherAPI, optional ElevenLabs/AWS). Then:

```
python main.py                          # main assistant
python IseeYou/GPUserver.py             # vision server
python interfaces/Whisper.py            # local Whisper STT
```

## Author

Felix Wa Ngoy Nsenga - fnsenga@seattleu.edu
