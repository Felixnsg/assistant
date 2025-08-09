# CLAUDE.md - Orpheus TTS Testing Project

## Project Overview
Testing suite for Orpheus TTS model deployment on vast.ai with local performance evaluation. Focus on streaming latency, voice cloning, emotion control, and quality benchmarking.

## Context Files
Read these files for complete project understanding:
- @paste.txt - Detailed test session plan with code examples and benchmarks
- All files in current directory for implementation details

## Tech Stack
- **Model**: Orpheus TTS (canopylabs/orpheus-tts-0.1-finetune-prod)
- **Python**: 3.8+ with CUDA support
- **Dependencies**: torch, vllm==0.7.3, orpheus-speech, soundfile, scipy
- **Deployment**: vast.ai RTX 4090 instance
- **Testing**: Local Python scripts + Flask streaming server

## Project Structure
- `test_basic.py` - Basic inference and latency tests
- `streaming_server.py` - Flask server for real-time audio streaming
- `test_voice_clone.py` - Voice cloning with audio samples
- `test_emotions.py` - Emotion tag testing (<laugh>, <sigh>, etc.)
- `benchmark.py` - Performance metrics (TTFB, total time)
- `my_voice.wav` - Voice cloning sample (10-30 seconds)
- `output/` - Generated audio files

## Key Commands
- `python test_basic.py` - Run basic TTS generation test
- `python streaming_server.py` - Start streaming server on port 8080
- `python benchmark.py` - Run performance benchmarks
- `python test_emotions.py` - Test emotion controls
- `curl "http://localhost:8080/tts?text=Hello&voice=tara"` - Test streaming endpoint

## Model Configuration
- **Sample Rate**: 24kHz output
- **Voices Available**: tara, leah, jess, leo, dan, mia, zac, zoe
- **Token Ratio**: ~83 tokens = 1 second audio
- **Memory Requirements**: 12GB+ VRAM for 3B model
- **Max Model Length**: 2048 tokens (reduce to 1024 if OOM)

## Performance Targets
- **TTFB (Time to First Byte)**: <500ms local, <1s remote
- **Streaming**: Native streaming support (not hacked)
- **Quality**: Natural prosody, clear speech
- **Stability**: No crashes or artifacts during generation

## Code Conventions
- Use descriptive variable names for audio processing
- Include timing measurements for all tests
- Save audio outputs with descriptive filenames
- Handle CUDA OOM gracefully with fallbacks
- Log first chunk timing separately from total time

## Troubleshooting
- **CUDA OOM**: Reduce max_model_len to 1024, clear GPU cache
- **Slow generation**: Enable torch.compile, check GPU utilization
- **VLLM errors**: Downgrade to vllm==0.7.3, verify CUDA compatibility
- **Poor quality**: Increase repetition_penalty to 1.3, adjust temperature

## Testing Workflow
1. Basic inference test - verify model loads and generates audio
2. Streaming test - confirm real-time audio delivery
3. Voice cloning test - test with personal audio sample
4. Emotion control test - verify emotion tags work
5. Performance benchmark - measure and document latency
6. Quality assessment - evaluate different voices and long-form content

## Success Criteria
- ✅ First audio chunk in <500ms
- ✅ Natural, clear speech quality
- ✅ Emotion tags produce expected effects
- ✅ Different voices sound distinct
- ✅ Stable generation without crashes
- ✅ Resource usage within acceptable limits

## Deployment Notes
- **vast.ai**: Rent RTX 4090, PyTorch 2.0+ image, open ports 8080,22
- **Local**: Requires 12GB+ VRAM, CUDA-capable GPU
- **Model Loading**: First load takes time, subsequent generations faster
- **Codec**: Uses SNAC (not DAC), 24kHz output