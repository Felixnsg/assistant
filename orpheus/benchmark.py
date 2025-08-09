#!/usr/bin/env python3
"""
Orpheus TTS Performance Benchmark
Comprehensive performance metrics and analysis
"""

import time
import wave
import torch
import os
import statistics
import json
from datetime import datetime
from orpheus_tts import OrpheusModel
import gc

class BenchmarkResults:
    """Store and analyze benchmark results"""
    
    def __init__(self):
        self.results = []
        
    def add_result(self, category, test_name, metrics):
        """Add a benchmark result"""
        self.results.append({
            "category": category,
            "test": test_name,
            "timestamp": datetime.now().isoformat(),
            **metrics
        })
    
    def print_summary(self):
        """Print benchmark summary"""
        print("\n" + "=" * 60)
        print("BENCHMARK SUMMARY")
        print("=" * 60)
        
        # Group by category
        categories = {}
        for r in self.results:
            cat = r["category"]
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(r)
        
        for category, results in categories.items():
            print(f"\n📊 {category}")
            print("-" * 40)
            
            # Calculate statistics
            ttfb_times = [r["ttfb_ms"] for r in results if "ttfb_ms" in r]
            total_times = [r["total_time_s"] for r in results if "total_time_s" in r]
            rtf_values = [r["rtf"] for r in results if "rtf" in r]
            
            if ttfb_times:
                print(f"  TTFB (ms):")
                print(f"    Min: {min(ttfb_times):.1f}")
                print(f"    Max: {max(ttfb_times):.1f}")
                print(f"    Mean: {statistics.mean(ttfb_times):.1f}")
                print(f"    Median: {statistics.median(ttfb_times):.1f}")
            
            if total_times:
                print(f"  Total Time (s):")
                print(f"    Min: {min(total_times):.2f}")
                print(f"    Max: {max(total_times):.2f}")
                print(f"    Mean: {statistics.mean(total_times):.2f}")
            
            if rtf_values:
                print(f"  Real-time Factor:")
                print(f"    Best: {min(rtf_values):.2f}x")
                print(f"    Worst: {max(rtf_values):.2f}x")
                print(f"    Mean: {statistics.mean(rtf_values):.2f}x")
    
    def save_json(self, filename="benchmark_results.json"):
        """Save results to JSON file"""
        with open(filename, "w") as f:
            json.dump(self.results, f, indent=2)
        print(f"\n💾 Results saved to {filename}")

def benchmark_performance():
    """Run comprehensive performance benchmarks"""
    
    print("=" * 60)
    print("ORPHEUS TTS PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # System info
    print("\n📋 System Information:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Initialize results tracker
    results = BenchmarkResults()
    
    # Model configuration
    model_name = "canopylabs/orpheus-tts-0.1-finetune-prod"
    max_model_len = 2048
    
    print(f"\n📦 Loading model: {model_name}")
    load_start = time.time()
    
    try:
        model = OrpheusModel(
            model_name=model_name,
            dtype=torch.bfloat16
        )
        load_time = time.time() - load_start
        print(f"✅ Model loaded in {load_time:.2f} seconds\n")
        
        results.add_result("Model Loading", "Initial Load", {
            "load_time_s": load_time
        })
        
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Test configurations
    test_configs = [
        # Length benchmarks
        {
            "category": "Text Length",
            "name": "Very Short (10 words)",
            "text": "Hello, this is a very short test of the TTS system.",
            "voice": "tara",
            "iterations": 3
        },
        {
            "category": "Text Length",
            "name": "Short (25 words)",
            "text": "The quick brown fox jumps over the lazy dog. This classic sentence is perfect for testing text-to-speech systems with various voices and settings.",
            "voice": "tara",
            "iterations": 3
        },
        {
            "category": "Text Length",
            "name": "Medium (50 words)",
            "text": "In the heart of the bustling city, where skyscrapers touched the clouds and streets hummed with endless activity, Sarah found a moment of peace in the small coffee shop on the corner. The aroma of freshly ground beans mixed with the sound of quiet conversations created a perfect escape.",
            "voice": "tara",
            "iterations": 3
        },
        {
            "category": "Text Length",
            "name": "Long (100 words)",
            "text": """The development of artificial intelligence has transformed numerous industries, from healthcare to transportation. 
            Machine learning algorithms now diagnose diseases with remarkable accuracy, while autonomous vehicles navigate complex urban environments. 
            Natural language processing enables seamless communication between humans and computers, breaking down language barriers worldwide. 
            Computer vision systems identify objects and patterns faster than human eyes. 
            However, these advancements also raise important ethical questions about privacy, employment, and decision-making. 
            As we continue to push the boundaries of what's possible, we must carefully consider the societal implications of these powerful technologies and ensure they benefit humanity.""",
            "voice": "tara",
            "iterations": 3
        },
        
        # Voice benchmarks
        {
            "category": "Voice Comparison",
            "name": "Tara",
            "text": "Testing voice generation speed and quality across different speaker models.",
            "voice": "tara",
            "iterations": 2
        },
        {
            "category": "Voice Comparison",
            "name": "Leo",
            "text": "Testing voice generation speed and quality across different speaker models.",
            "voice": "leo",
            "iterations": 2
        },
        {
            "category": "Voice Comparison",
            "name": "Mia",
            "text": "Testing voice generation speed and quality across different speaker models.",
            "voice": "mia",
            "iterations": 2
        },
        
        # Parameter variations
        {
            "category": "Temperature",
            "name": "Low (0.3)",
            "text": "Testing how temperature affects generation speed and quality.",
            "voice": "tara",
            "temperature": 0.3,
            "iterations": 2
        },
        {
            "category": "Temperature",
            "name": "Medium (0.6)",
            "text": "Testing how temperature affects generation speed and quality.",
            "voice": "tara",
            "temperature": 0.6,
            "iterations": 2
        },
        {
            "category": "Temperature",
            "name": "High (0.9)",
            "text": "Testing how temperature affects generation speed and quality.",
            "voice": "tara",
            "temperature": 0.9,
            "iterations": 2
        },
        
        # Emotion benchmarks
        {
            "category": "Emotions",
            "name": "With Tags",
            "text": "<laugh> This is so funny! <sigh> But also a bit tiring. <gasp> Wait, what was that?",
            "voice": "tara",
            "iterations": 2
        },
        {
            "category": "Emotions",
            "name": "Without Tags",
            "text": "This is so funny! But also a bit tiring. Wait, what was that?",
            "voice": "tara",
            "iterations": 2
        }
    ]
    
    # Run benchmarks
    for config in test_configs:
        print(f"\n🔄 Benchmarking: {config['category']} - {config['name']}")
        print(f"  Text length: {len(config['text'])} chars")
        print(f"  Iterations: {config['iterations']}")
        
        # Run multiple iterations
        iteration_results = []
        
        for i in range(config['iterations']):
            # Clear GPU cache between runs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
            
            # Get parameters
            temperature = config.get('temperature', 0.6)
            top_p = config.get('top_p', 0.8)
            repetition_penalty = config.get('repetition_penalty', 1.3)
            
            # Measure generation
            start_time = time.time()
            first_chunk_time = None
            
            try:
                syn_tokens = model.generate_speech(
                    prompt=config['text'],
                    voice=config['voice'],
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=2000,
                    repetition_penalty=repetition_penalty,
                    stop_token_ids=[128258]
                )
                
                # Collect audio without saving (for speed)
                audio_buffer = []
                chunk_count = 0
                
                for audio_chunk in syn_tokens:
                    if first_chunk_time is None:
                        first_chunk_time = time.time() - start_time
                    
                    chunk_count += 1
                    audio_buffer.append(audio_chunk)
                
                total_time = time.time() - start_time
                
                # Calculate audio duration
                total_bytes = sum(len(chunk) for chunk in audio_buffer)
                total_frames = total_bytes // 2  # 16-bit samples
                duration = total_frames / 24000  # 24kHz sample rate
                
                # Calculate metrics
                ttfb_ms = first_chunk_time * 1000
                rtf = total_time / duration if duration > 0 else 0
                tokens_per_second = chunk_count / total_time if total_time > 0 else 0
                
                iteration_results.append({
                    "ttfb_ms": ttfb_ms,
                    "total_time_s": total_time,
                    "audio_duration_s": duration,
                    "rtf": rtf,
                    "chunks": chunk_count,
                    "tokens_per_second": tokens_per_second
                })
                
                print(f"  Run {i+1}: TTFB={ttfb_ms:.1f}ms, Time={total_time:.2f}s, RTF={rtf:.2f}x")
                
            except Exception as e:
                print(f"  Run {i+1}: ❌ Failed - {e}")
        
        # Calculate statistics for this test
        if iteration_results:
            avg_ttfb = statistics.mean([r["ttfb_ms"] for r in iteration_results])
            avg_time = statistics.mean([r["total_time_s"] for r in iteration_results])
            avg_rtf = statistics.mean([r["rtf"] for r in iteration_results])
            
            results.add_result(config['category'], config['name'], {
                "text_length": len(config['text']),
                "voice": config['voice'],
                "iterations": config['iterations'],
                "ttfb_ms": avg_ttfb,
                "total_time_s": avg_time,
                "rtf": avg_rtf,
                "temperature": temperature,
                "all_iterations": iteration_results
            })
            
            print(f"  📊 Average: TTFB={avg_ttfb:.1f}ms, RTF={avg_rtf:.2f}x")
    
    # Memory usage benchmark
    if torch.cuda.is_available():
        print("\n🔄 Memory Usage Benchmark")
        print("-" * 40)
        
        # Get baseline memory
        torch.cuda.empty_cache()
        gc.collect()
        baseline_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"  Baseline: {baseline_memory:.2f} GB")
        
        # Generate with long text
        long_text = "This is a test. " * 100  # Very long text
        
        try:
            syn_tokens = model.generate_speech(
                prompt=long_text,
                voice="tara",
                max_tokens=2000
            )
            
            # Consume generator
            for _ in syn_tokens:
                pass
            
            peak_memory = torch.cuda.max_memory_allocated() / 1024**3
            current_memory = torch.cuda.memory_allocated() / 1024**3
            
            print(f"  Peak: {peak_memory:.2f} GB")
            print(f"  Current: {current_memory:.2f} GB")
            
            results.add_result("Memory", "VRAM Usage", {
                "baseline_gb": baseline_memory,
                "peak_gb": peak_memory,
                "final_gb": current_memory
            })
            
        except Exception as e:
            print(f"  ❌ Memory test failed: {e}")
    
    # Print summary
    results.print_summary()
    
    # Save results
    results.save_json("output/benchmark_results.json")
    
    # Performance assessment
    print("\n" + "=" * 60)
    print("PERFORMANCE ASSESSMENT")
    print("=" * 60)
    
    # Check against targets
    all_ttfb = [r["ttfb_ms"] for r in results.results if "ttfb_ms" in r]
    if all_ttfb:
        avg_ttfb = statistics.mean(all_ttfb)
        min_ttfb = min(all_ttfb)
        
        print(f"\n✅ TTFB Target (<500ms): {'PASS' if avg_ttfb < 500 else 'FAIL'}")
        print(f"   Best: {min_ttfb:.1f}ms")
        print(f"   Average: {avg_ttfb:.1f}ms")
    
    all_rtf = [r["rtf"] for r in results.results if "rtf" in r]
    if all_rtf:
        avg_rtf = statistics.mean(all_rtf)
        
        print(f"\n✅ Real-time Factor: {'PASS' if avg_rtf < 1.0 else 'MARGINAL'}")
        print(f"   Average RTF: {avg_rtf:.2f}x")
        print(f"   (Lower is better, <1.0 means faster than real-time)")
    
    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    # Create output directory
    os.makedirs("output", exist_ok=True)
    
    # Run benchmark
    benchmark_performance()