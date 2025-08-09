#!/usr/bin/env python3
"""
Orpheus TTS Emotion Control Test
Tests emotion tags and expressive speech generation
"""

import time
import wave
import torch
import os
from orpheus_tts import OrpheusModel

def test_emotions():
    print("=" * 60)
    print("ORPHEUS TTS EMOTION CONTROL TEST")
    print("=" * 60)
    
    # Initialize model
    print(f"\n📦 Loading model...")
    model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
    
    try:
        model = OrpheusModel(
            model_name=model_name,
            dtype=torch.bfloat16
        )
        print(f"✅ Model loaded\n")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Emotion tag tests
    emotion_tests = [
        {
            "name": "Laughter",
            "prompt": "Oh my goodness, <laugh> that's the funniest thing I've heard all day! <laugh> I can't believe that actually happened!",
            "voice": "tara",
            "description": "Natural laughter in speech"
        },
        {
            "name": "Chuckling",
            "prompt": "Well, <chuckle> I guess we all make mistakes sometimes. <chuckle> At least we can laugh about it now.",
            "voice": "leo",
            "description": "Soft chuckling"
        },
        {
            "name": "Sighing",
            "prompt": "<sigh> I've been working on this project all day. <sigh> Sometimes I wonder if it's worth all the effort.",
            "voice": "leah",
            "description": "Tired/frustrated sighs"
        },
        {
            "name": "Coughing",
            "prompt": "Sorry, I've been feeling a bit under the weather. <cough> <cough> Let me clear my throat and continue.",
            "voice": "dan",
            "description": "Natural coughing sounds"
        },
        {
            "name": "Sniffling",
            "prompt": "It's just... <sniffle> I didn't expect things to turn out this way. <sniffle> I'm trying to stay strong.",
            "voice": "mia",
            "description": "Emotional sniffling"
        },
        {
            "name": "Groaning",
            "prompt": "<groan> Not another meeting! <groan> My calendar is already completely full today.",
            "voice": "zac",
            "description": "Frustrated groaning"
        },
        {
            "name": "Yawning",
            "prompt": "<yawn> Sorry, I stayed up way too late last night. <yawn> I really need some coffee.",
            "voice": "jess",
            "description": "Tired yawning"
        },
        {
            "name": "Gasping",
            "prompt": "<gasp> No way! <gasp> Did that really just happen? I can't believe my eyes!",
            "voice": "zoe",
            "description": "Surprised gasping"
        },
        {
            "name": "Mixed emotions",
            "prompt": "<gasp> Wait, you're serious? <laugh> That's incredible! <sigh> Though I wish you'd told me sooner. <chuckle> Better late than never, I suppose!",
            "voice": "tara",
            "description": "Multiple emotions in one utterance"
        }
    ]
    
    # Additional emotion styles (from emotions.txt)
    emotion_styles = [
        {
            "name": "Happy",
            "prompt": "This is absolutely wonderful news! I'm so thrilled to hear about your success!",
            "voice": "tara",
            "style": "happy"
        },
        {
            "name": "Sad",
            "prompt": "I'm really sorry to hear that. It must be so difficult for you right now.",
            "voice": "leah",
            "style": "sad"
        },
        {
            "name": "Angry",
            "prompt": "This is completely unacceptable! I demand an explanation right now!",
            "voice": "leo",
            "style": "angry"
        },
        {
            "name": "Whisper",
            "prompt": "Shh, we need to be very quiet. I don't want anyone to hear us talking about this.",
            "voice": "mia",
            "style": "whisper"
        },
        {
            "name": "Excited",
            "prompt": "Oh wow, this is amazing! I can't wait to tell everyone about this incredible opportunity!",
            "voice": "jess",
            "style": "excited"
        },
        {
            "name": "Sleepy",
            "prompt": "Mmm, I'm so tired... I can barely keep my eyes open... Just five more minutes...",
            "voice": "dan",
            "style": "sleepy"
        }
    ]
    
    print("🎭 Testing emotion tags...")
    print("-" * 40)
    
    # Test emotion tags
    for i, test in enumerate(emotion_tests, 1):
        print(f"\nTest {i}: {test['name']}")
        print(f"  Description: {test['description']}")
        print(f"  Voice: {test['voice']}")
        
        start_time = time.time()
        first_chunk_time = None
        
        try:
            syn_tokens = model.generate_speech(
                prompt=test['prompt'],
                voice=test['voice'],
                temperature=0.7,  # Slightly higher for more expression
                top_p=0.85,
                max_tokens=1500,
                repetition_penalty=1.2,
                stop_token_ids=[128258]
            )
            
            output_filename = f"output/emotion_{i}_{test['name'].lower().replace(' ', '_')}.wav"
            
            with wave.open(output_filename, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(24000)
                
                total_frames = 0
                
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
                print(f"  💾 Saved to: {output_filename}")
                
        except Exception as e:
            print(f"  ❌ Generation failed: {e}")
    
    print("\n" + "=" * 40)
    print("🎨 Testing emotion styles...")
    print("-" * 40)
    
    # Test emotion styles
    for i, test in enumerate(emotion_styles, 1):
        print(f"\nStyle Test {i}: {test['style'].upper()}")
        print(f"  Voice: {test['voice']}")
        
        start_time = time.time()
        
        try:
            # Adjust parameters based on emotion style
            temp_adjust = {
                "happy": 0.7,
                "sad": 0.5,
                "angry": 0.8,
                "whisper": 0.4,
                "excited": 0.9,
                "sleepy": 0.3
            }
            
            temperature = temp_adjust.get(test['style'], 0.6)
            
            syn_tokens = model.generate_speech(
                prompt=test['prompt'],
                voice=test['voice'],
                temperature=temperature,
                top_p=0.85,
                max_tokens=1200,
                repetition_penalty=1.3,
                stop_token_ids=[128258]
            )
            
            output_filename = f"output/style_{test['style']}.wav"
            
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
    
    print("\n" + "=" * 60)
    print("EMOTION CONTROL TEST COMPLETE")
    print(f"Generated {len(emotion_tests)} emotion tag samples")
    print(f"Generated {len(emotion_styles)} emotion style samples")
    print("Check output/ directory for all generated files")
    print("=" * 60)

if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run test
    test_emotions()