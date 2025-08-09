# ORPHEUS TTS STANDALONE TEST SESSION

## Objective
Test Orpheus TTS as a standalone system before integrating into Cypher. We'll verify streaming, voice cloning, and emotion control work as advertised.

## Session Goals
1. **Basic Setup** - Get Orpheus running locally or on vast.ai
2. **Test Streaming** - Verify real-time audio generation
3. **Voice Cloning** - Test with 10-30 second samples
4. **Emotion Tags** - Test `<laugh>`, `<sigh>`, emotion states
5. **Performance Baseline** - Measure latency and quality

## Prerequisites Check
- Python 3.8+ environment
- CUDA-capable GPU (local) OR vast.ai account
- ~15GB free space for model
- Basic audio recording for voice samples

## Step 1: Environment Setup

### Local Setup (if 12GB+ VRAM)
```bash
# Create fresh environment
python -m venv orpheus_test
source orpheus_test/bin/activate  # or `orpheus_test\Scripts\activate` on Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install vllm==0.7.3  # Specific version to avoid bugs
pip install orpheus-speech
pip install soundfile scipy
```

### vast.ai Setup (if no local GPU)
1. Rent RTX 4090 instance (~$0.40/hr)
2. Select PyTorch 2.0+ image
3. Open ports 8080, 22
4. SSH in and run setup commands above

## Step 2: Basic Inference Test

Create `test_basic.py`:
```python
from orpheus_tts import OrpheusModel
import soundfile as sf
import time

# Load model
print("Loading Orpheus model...")
model = OrpheusModel(
    model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
    max_model_len=2048
)

# Simple test
text = "Hello world! This is a test of Orpheus text to speech."
print(f"Generating speech for: {text}")

start = time.time()
audio_generator = model.generate_speech(
    prompt=text,
    voice="tara",  # Available: tara, leah, jess, leo, dan, mia, zac, zoe
    temperature=0.6,
    repetition_penalty=1.1
)

# Collect audio chunks
audio_data = []
chunk_count = 0
for chunk in audio_generator:
    chunk_count += 1
    audio_data.append(chunk)
    if chunk_count == 1:
        print(f"First chunk received in {time.time()-start:.2f}s")

# Save complete audio
full_audio = b''.join(audio_data)
with open("test_output.wav", "wb") as f:
    f.write(full_audio)

print(f"Total generation time: {time.time()-start:.2f}s")
print(f"Audio saved to test_output.wav")
```

**Expected Results:**
- First chunk in <500ms (local) or <1s (remote)
- Natural sounding voice
- Complete audio file

## Step 3: Streaming Server Test

Create `streaming_server.py`:
```python
from flask import Flask, Response, request
from orpheus_tts import OrpheusModel
import struct

app = Flask(__name__)
model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")

def create_wav_header(sample_rate=24000):
    # WAV header for streaming
    return struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', 36, b'WAVE', b'fmt ', 16, 1, 1,
        sample_rate, sample_rate * 2, 2, 16, b'data', 0
    )

@app.route('/tts')
def tts():
    text = request.args.get('text', 'Hello from Orpheus!')
    voice = request.args.get('voice', 'tara')
    
    def generate():
        yield create_wav_header()
        for chunk in model.generate_speech(prompt=text, voice=voice):
            yield chunk
    
    return Response(generate(), mimetype='audio/wav')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

**Test with:**
- Browser: `http://localhost:8080/tts?text=Hello%20world&voice=tara`
- Should stream audio immediately

## Step 4: Voice Cloning Test

1. **Record your voice sample** (10-30 seconds):
   - Clear speech, normal pace
   - Save as `my_voice.wav`

2. Create `test_voice_clone.py`:
```python
# Note: Voice cloning requires pretrained model
model = OrpheusModel(
    model_name="canopylabs/orpheus-tts-0.1-pretrained",  # Pretrained for cloning
    max_model_len=4096
)

# Format: Provide audio context in prompt
clone_prompt = """[Audio: my_voice.wav]
Now I will speak in the cloned voice: Hello, this is a test of voice cloning."""

audio = model.generate_speech(
    prompt=clone_prompt,
    temperature=0.7,
    repetition_penalty=1.2
)
```

## Step 5: Emotion Control Test

Create `test_emotions.py`:
```python
emotion_tests = [
    "I'm so excited about this! <laugh>",
    "This is really sad news. <sigh>",
    "WHAT?! I can't believe it! (shouting)",
    "Let me tell you a secret... (whispering)",
    "I'm feeling very tired today. <yawn>",
    "That's absolutely hilarious! <chuckle>",
]

for i, text in enumerate(emotion_tests):
    print(f"Generating: {text}")
    audio = model.generate_speech(prompt=text, voice="tara")
    # Save each emotion
    save_audio(audio, f"emotion_{i}.wav")
```

## Step 6: Performance Benchmarks

Create `benchmark.py`:
```python
import time
import statistics

def benchmark_latency(model, text, runs=5):
    ttfb_times = []  # Time to first byte
    total_times = []
    
    for _ in range(runs):
        start = time.time()
        first_chunk_time = None
        
        for chunk in model.generate_speech(prompt=text, voice="tara"):
            if first_chunk_time is None:
                first_chunk_time = time.time() - start
                ttfb_times.append(first_chunk_time)
        
        total_times.append(time.time() - start)
    
    print(f"TTFB Average: {statistics.mean(ttfb_times)*1000:.0f}ms")
    print(f"Total Average: {statistics.mean(total_times):.2f}s")
    print(f"TTFB Range: {min(ttfb_times)*1000:.0f}-{max(ttfb_times)*1000:.0f}ms")
```

## Step 7: Quality Tests

### Test Different Voices
```python
voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
test_text = "This is a test of different voice personalities."

for voice in voices:
    audio = model.generate_speech(prompt=test_text, voice=voice)
    save_audio(audio, f"voice_{voice}.wav")
```

### Test Long-form Content
```python
long_text = """
This is a longer passage to test how Orpheus handles extended speech synthesis.
We want to see if the quality remains consistent, if the streaming works smoothly,
and whether there are any artifacts or issues with longer generation.
The model should maintain natural prosody and intonation throughout.
"""
```

## Troubleshooting Guide

### Common Issues:

1. **CUDA out of memory**
   - Reduce `max_model_len` to 1024
   - Use smaller batch size
   - Clear GPU cache

2. **Slow generation**
   - Enable torch.compile: `use_torch_compile=True`
   - Check GPU utilization
   - Reduce temperature

3. **Poor audio quality**
   - Increase `repetition_penalty` to 1.3
   - Adjust temperature (0.4-0.8)
   - Try different voices

4. **VLLM errors**
   - Downgrade to `vllm==0.7.3`
   - Check CUDA version compatibility

## Success Criteria

✅ **Streaming works** - First audio in <500ms
✅ **Quality is good** - Natural, clear speech
✅ **Emotions work** - Tags produce expected effects
✅ **Voice variety** - Different voices sound distinct
✅ **Stable generation** - No crashes or artifacts

## Next Steps

Once all tests pass:
1. Document optimal settings
2. Measure resource usage
3. Test edge cases
4. Plan Cypher integration
5. Design voice cloning workflow

## Notes

- Orpheus uses SNAC codec (not DAC)
- 24kHz sample rate output
- ~83 tokens = 1 second of audio
- 3B model needs ~12GB VRAM
- Streaming is native, not hacked

## Integration Readiness Checklist

- [ ] Basic generation works
- [ ] Streaming confirmed
- [ ] Latency acceptable (<500ms)
- [ ] Voice quality good
- [ ] Emotion control works
- [ ] Resource usage reasonable
- [ ] No blocking issues

Once this checklist is complete, we're ready to integrate Orpheus into Cypher's pipeline!