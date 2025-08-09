#!/usr/bin/env python3
"""
Basic Orpheus TTS inference test
Tests model loading, basic generation, and saves output
"""

import time
import wave
import torch
from orpheus_tts import OrpheusModel

def test_basic_inference():
    print("=" * 60)
    print("ORPHEUS TTS BASIC INFERENCE TEST")
    print("=" * 60)
    
    # Model configuration
    model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
    max_model_len = 2048  # Reduce to 1024 if OOM
    
    print(f"\n📦 Loading model: {model_name}")
    print(f"   Max model length: {max_model_len}")
    
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    device = "cuda" if cuda_available else "cpu"
    print(f"   Device: {device}")
    if cuda_available:
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize model
    load_start = time.time()
    try:
        model = OrpheusModel(
            model_name=model_name,
            dtype=torch.bfloat16,
            max_model_len=max_model_len
        )
        load_time = time.time() - load_start
        print(f"✅ Model loaded in {load_time:.2f} seconds\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test prompts
    test_cases = [
        {
            "name": "Short greeting",
            "prompt": "Hello there! Welcome to Orpheus TTS testing.",
            "voice": "tara"
        },
        {
            "name": "Medium conversation",
            "prompt": "Hey, so I was thinking about what we discussed yesterday, and honestly, I think you might be onto something. The whole approach just makes way more sense when you look at it from that perspective.",
            "voice": "leo"
        },
        {
            "name": "Long narrative",
            "prompt": """The sun was setting over the quiet town, casting long shadows across the empty streets. 
            Sarah walked slowly, her footsteps echoing in the silence. She couldn't shake the feeling that something was different today, 
            something in the air that made everything feel just a little bit off. As she turned the corner, she saw it - 
            the old bookstore that had been closed for years was suddenly open, its windows glowing with warm light.""",
            "voice": "leah"
        }
    ]
    
    # Run tests
    for i, test in enumerate(test_cases, 1):
        print("-" * 40)
        print(f"Test {i}: {test['name']}")
        print(f"Voice: {test['voice']}")
        print(f"Text length: {len(test['prompt'])} chars")
        
        # Measure generation time
        gen_start = time.time()
        first_chunk_time = None
        
        try:
            # Generate speech
            syn_tokens = model.generate_speech(
                prompt=test['prompt'],
                voice=test['voice'],
                temperature=0.6,
                top_p=0.8,
                max_tokens=1200,
                repetition_penalty=1.3,
                stop_token_ids=[128258]
            )
            
            # Save to file
            output_filename = f"output/test_{i}_{test['voice']}.wav"
            
            with wave.open(output_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                total_frames = 0
                chunk_count = 0
                
                for audio_chunk in syn_tokens:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - gen_start
                        print(f"  ⏱️ Time to first chunk: {first_chunk_time*1000:.1f}ms")
                    
                    chunk_count += 1
                    frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                    total_frames += frame_count
                    wf.writeframes(audio_chunk)
                
                # Calculate statistics
                total_time = time.time() - gen_start
                duration = total_frames / wf.getframerate()
                rtf = total_time / duration if duration > 0 else 0
                
                print(f"  ✅ Generated {duration:.2f}s of audio in {total_time:.2f}s")
                print(f"  📊 Real-time factor: {rtf:.2f}x")
                print(f"  📊 Chunks generated: {chunk_count}")
                print(f"  💾 Saved to: {output_filename}")
                
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
            if "out of memory" in str(e).lower():
                print("  💡 Try reducing max_model_len to 1024")
    
    print("\n" + "=" * 60)
    print("BASIC INFERENCE TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    # Create output directory
    import os
    os.makedirs("output", exist_ok=True)
    
    # Run test
    test_basic_inference()