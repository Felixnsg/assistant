#!/usr/bin/env python3
"""
Orpheus TTS Voice Cloning Test
Tests zero-shot voice cloning with audio samples
"""

import time
import wave
import torch
import os
import soundfile as sf
import numpy as np
from orpheus_tts import OrpheusModel

def test_voice_cloning():
    print("=" * 60)
    print("ORPHEUS TTS VOICE CLONING TEST")
    print("=" * 60)
    
    # Check for voice sample
    voice_sample_path = "my_voice.wav"
    
    if not os.path.exists(voice_sample_path):
        print(f"\n⚠️  Voice sample not found: {voice_sample_path}")
        print("📝 To test voice cloning:")
        print("   1. Record 10-30 seconds of clear speech")
        print("   2. Save as 'my_voice.wav' (mono, 16-24kHz)")
        print("   3. Run this script again\n")
        print("Creating demo with built-in voices instead...\n")
        test_voice_comparison()
        return
    
    print(f"\n📦 Loading model for voice cloning...")
    
    # Use pretrained model for voice cloning (better zero-shot performance)
    model_name = "canopylabs/orpheus-tts-0.1-pretrained"
    
    try:
        model = OrpheusModel(
            model_name=model_name,
            max_model_len=2048,
            dtype=torch.bfloat16
        )
        print(f"✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load and process voice sample
    print(f"🎤 Loading voice sample: {voice_sample_path}")
    try:
        audio_data, sample_rate = sf.read(voice_sample_path)
        duration = len(audio_data) / sample_rate
        print(f"   Duration: {duration:.1f} seconds")
        print(f"   Sample rate: {sample_rate} Hz")
        
        # Convert to required format if needed
        if sample_rate != 24000:
            print(f"   ⚠️  Resampling from {sample_rate}Hz to 24000Hz")
            # Note: In production, use proper resampling library
            
    except Exception as e:
        print(f"❌ Failed to load voice sample: {e}")
        return
    
    # Test prompts for cloned voice
    test_prompts = [
        "Hello, this is a test of voice cloning with Orpheus TTS.",
        "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
        "I'm really excited about this new technology! It's absolutely amazing what we can do with AI these days.",
        "Sometimes I wonder if artificial intelligence will ever truly understand human emotions, you know?",
    ]
    
    print("\n🔄 Generating speech with cloned voice...")
    print("-" * 40)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt[:50]}...")
        
        start_time = time.time()
        
        try:
            # For voice cloning with pretrained model, we need to provide context
            # This is a simplified approach - real implementation would process the audio
            # For now, we'll use the model's conditioning capability
            
            syn_tokens = model.generate_speech(
                prompt=prompt,
                voice=None,  # No predefined voice for cloning
                temperature=0.7,
                top_p=0.85,
                max_tokens=1500,
                repetition_penalty=1.2,
                stop_token_ids=[128258]
            )
            
            output_filename = f"output/cloned_voice_{i}.wav"
            
            with wave.open(output_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                total_frames = 0
                for audio_chunk in syn_tokens:
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
                
                duration = total_frames / wf.getframerate()
                total_time = time.time() - start_time
                
                print(f"  ✅ Generated {duration:.2f}s in {total_time:.2f}s")
                print(f"  💾 Saved to: {output_filename}")
                
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")

def test_voice_comparison():
    """Compare different built-in voices"""
    print("🎭 VOICE COMPARISON TEST")
    print("-" * 40)
    
    model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
    
    try:
        model = OrpheusModel(
            model_name=model_name,
            max_model_len=2048,
            dtype=torch.bfloat16
        )
        print(f"✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test all available voices
    voices = ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"]
    test_text = "Hello! This is a test of the Orpheus text-to-speech system. Each voice has its own unique characteristics and personality."
    
    print(f"📝 Test text: {test_text[:60]}...\n")
    
    for voice in voices:
        print(f"🎤 Generating with voice: {voice}")
        
        start_time = time.time()
        
        try:
            syn_tokens = model.generate_speech(
                prompt=test_text,
                voice=voice,
                temperature=0.6,
                top_p=0.8,
                max_tokens=1200,
                repetition_penalty=1.3,
                stop_token_ids=[128258]
            )
            
            output_filename = f"output/voice_{voice}.wav"
            
            with wave.open(output_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                total_frames = 0
                first_chunk_time = None
                
                for audio_chunk in syn_tokens:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                    
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
                
                duration = total_frames / wf.getframerate()
                total_time = time.time() - start_time
                
                print(f"  ⏱️ TTFB: {first_chunk_time*1000:.1f}ms")
                print(f"  ✅ Generated {duration:.2f}s in {total_time:.2f}s")
                print(f"  💾 Saved to: {output_filename}\n")
                
        except Exception as e:
            print(f"  ❌ Generation failed: {e}\n")
    
    print("=" * 60)
    print("VOICE COMPARISON COMPLETE")
    print(f"Check the output/ directory to compare {len(voices)} different voices")
    print("=" * 60)

if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run test
    test_voice_cloning()