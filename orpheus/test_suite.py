#!/usr/bin/env python3
"""
Orpheus TTS Test Suite - Comprehensive testing for all features
Consolidated from multiple test files into one organized suite
"""

import time
import requests
import soundfile as sf
import numpy as np
import json
import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
import asyncio
import websockets
import base64
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OrpheusTestSuite:
    """Comprehensive test suite for Orpheus TTS"""
    
    def __init__(self, server_url: str = "http://localhost:8080"):
        self.server_url = server_url.rstrip('/')
        self.results = []
        self.output_dir = Path("test_output")
        self.output_dir.mkdir(exist_ok=True)
        
    def check_server_health(self) -> bool:
        """Check if server is healthy"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def test_basic_generation(self) -> Dict[str, Any]:
        """Test basic TTS generation"""
        logger.info("\n=== Testing Basic Generation ===")
        
        test_text = "Hello, this is a test of the Orpheus text to speech system."
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.server_url}/tts",
                json={
                    "text": test_text,
                    "voice": "tara",
                    "temperature": 0.6,
                    "top_p": 0.8,
                    "max_tokens": 1000
                },
                stream=True
            )
            
            # Measure TTFB
            first_byte_time = None
            audio_data = b""
            
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    if first_byte_time is None:
                        first_byte_time = time.time() - start_time
                    audio_data += chunk
            
            total_time = time.time() - start_time
            
            # Save audio
            output_file = self.output_dir / "test_basic.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            
            # Analyze audio
            data, sr = sf.read(output_file)
            duration = len(data) / sr
            
            result = {
                "test": "basic_generation",
                "status": "PASS",
                "ttfb_ms": first_byte_time * 1000,
                "total_time_s": total_time,
                "audio_duration_s": duration,
                "rtf": total_time / duration if duration > 0 else 0,
                "file": str(output_file)
            }
            
            logger.info(f"✅ Basic generation successful")
            logger.info(f"   TTFB: {result['ttfb_ms']:.1f}ms")
            logger.info(f"   Audio duration: {duration:.1f}s")
            logger.info(f"   RTF: {result['rtf']:.2f}x")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Basic generation failed: {e}")
            return {"test": "basic_generation", "status": "FAIL", "error": str(e)}
    
    def test_long_generation(self) -> Dict[str, Any]:
        """Test long-form content generation with max_tokens=64000"""
        logger.info("\n=== Testing Long Generation (64000 tokens) ===")
        
        # Create a long text (about 2 minutes worth)
        long_text = """
        Once upon a time, in a distant galaxy, there lived a civilization that had mastered the art of 
        interstellar travel. Their ships could traverse the vast emptiness of space in mere moments, 
        bending the fabric of spacetime itself. This is their story, a tale of adventure, discovery, 
        and the eternal quest for knowledge that drives all sentient beings.
        
        The journey began on a small planet called Kepler-438b, where the first signs of this advanced 
        technology were discovered. Ancient ruins, buried deep beneath layers of cosmic dust, held 
        secrets that would change the course of history. Scientists worked tirelessly to decode the 
        mysterious symbols etched into crystalline structures that seemed to pulse with an otherworldly 
        energy.
        """ * 10  # Repeat to make it longer
        
        try:
            start_time = time.time()
            
            response = requests.post(
                f"{self.server_url}/tts",
                json={
                    "text": long_text,
                    "voice": "tara",
                    "max_tokens": 64000  # Test the full capacity!
                },
                stream=True,
                timeout=300  # 5 minute timeout for long generation
            )
            
            # Check headers
            max_tokens = response.headers.get('X-Max-Tokens', 'unknown')
            max_duration = response.headers.get('X-Max-Duration-Seconds', 'unknown')
            
            logger.info(f"   Server reports max_tokens: {max_tokens}")
            logger.info(f"   Server reports max_duration: {max_duration}s")
            
            # Stream and save
            audio_data = b""
            chunk_count = 0
            
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    audio_data += chunk
                    chunk_count += 1
                    if chunk_count % 100 == 0:
                        logger.info(f"   Received {len(audio_data) / 1024 / 1024:.1f}MB...")
            
            total_time = time.time() - start_time
            
            # Save audio
            output_file = self.output_dir / "test_long.wav"
            with open(output_file, 'wb') as f:
                f.write(audio_data)
            
            # Analyze audio
            data, sr = sf.read(output_file)
            duration = len(data) / sr
            
            result = {
                "test": "long_generation",
                "status": "PASS",
                "total_time_s": total_time,
                "audio_duration_s": duration,
                "audio_size_mb": len(audio_data) / 1024 / 1024,
                "max_tokens_used": max_tokens,
                "file": str(output_file)
            }
            
            logger.info(f"✅ Long generation successful")
            logger.info(f"   Generated {duration:.1f}s of audio ({duration/60:.1f} minutes)")
            logger.info(f"   File size: {result['audio_size_mb']:.1f}MB")
            logger.info(f"   Generation time: {total_time:.1f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Long generation failed: {e}")
            return {"test": "long_generation", "status": "FAIL", "error": str(e)}
    
    def test_different_voices(self) -> Dict[str, Any]:
        """Test all available voices"""
        logger.info("\n=== Testing Different Voices ===")
        
        voices = ["tara", "leah", "leo", "mia", "dan"]
        test_text = "This is a test of the voice synthesis capabilities."
        results = {}
        
        for voice in voices:
            try:
                response = requests.post(
                    f"{self.server_url}/tts",
                    json={
                        "text": test_text,
                        "voice": voice,
                        "max_tokens": 500
                    }
                )
                
                if response.status_code == 200:
                    # Save audio
                    output_file = self.output_dir / f"test_voice_{voice}.wav"
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    results[voice] = "PASS"
                    logger.info(f"✅ Voice {voice}: Success")
                else:
                    results[voice] = f"FAIL: {response.status_code}"
                    logger.error(f"❌ Voice {voice}: Failed with status {response.status_code}")
                    
            except Exception as e:
                results[voice] = f"FAIL: {str(e)}"
                logger.error(f"❌ Voice {voice}: {e}")
        
        return {"test": "different_voices", "results": results}
    
    def test_emotion_tags(self) -> Dict[str, Any]:
        """Test emotion tag support"""
        logger.info("\n=== Testing Emotion Tags ===")
        
        emotion_tests = [
            ("Happy laugh", "That's hilarious! <laugh> I can't believe it!"),
            ("Sigh", "This is exhausting... <sigh> Let me think about it."),
            ("Multiple emotions", "Oh wow! <gasp> That's amazing! <laugh> But also <sigh> quite tiring."),
            ("Chuckle", "Well, <chuckle> that's certainly one way to do it."),
        ]
        
        results = {}
        
        for name, text in emotion_tests:
            try:
                response = requests.post(
                    f"{self.server_url}/tts",
                    json={
                        "text": text,
                        "voice": "tara",
                        "temperature": 0.7,  # Slightly higher for emotions
                        "max_tokens": 1000
                    }
                )
                
                if response.status_code == 200:
                    output_file = self.output_dir / f"test_emotion_{name.replace(' ', '_').lower()}.wav"
                    with open(output_file, 'wb') as f:
                        f.write(response.content)
                    
                    results[name] = "PASS"
                    logger.info(f"✅ Emotion '{name}': Success")
                else:
                    results[name] = f"FAIL: {response.status_code}"
                    logger.error(f"❌ Emotion '{name}': Failed")
                    
            except Exception as e:
                results[name] = f"FAIL: {str(e)}"
                logger.error(f"❌ Emotion '{name}': {e}")
        
        return {"test": "emotion_tags", "results": results}
    
    async def test_websocket_streaming(self) -> Dict[str, Any]:
        """Test WebSocket streaming"""
        logger.info("\n=== Testing WebSocket Streaming ===")
        
        ws_url = self.server_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        
        try:
            async with websockets.connect(ws_url) as websocket:
                # Send request
                await websocket.send(json.dumps({
                    "text": "Testing WebSocket streaming functionality.",
                    "voice": "tara",
                    "max_tokens": 1000
                }))
                
                chunks_received = 0
                audio_data = b""
                
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get("type") == "audio_chunk":
                        chunks_received += 1
                        # Decode base64 audio
                        chunk_data = base64.b64decode(data["data"])
                        audio_data += chunk_data
                    elif data.get("type") == "complete":
                        break
                    elif data.get("error"):
                        raise Exception(data["error"])
                
                # Save audio
                output_file = self.output_dir / "test_websocket.wav"
                with open(output_file, 'wb') as f:
                    f.write(audio_data)
                
                result = {
                    "test": "websocket_streaming",
                    "status": "PASS",
                    "chunks_received": chunks_received,
                    "audio_size_bytes": len(audio_data),
                    "file": str(output_file)
                }
                
                logger.info(f"✅ WebSocket streaming successful")
                logger.info(f"   Received {chunks_received} chunks")
                logger.info(f"   Total size: {len(audio_data) / 1024:.1f}KB")
                
                return result
                
        except Exception as e:
            logger.error(f"❌ WebSocket streaming failed: {e}")
            return {"test": "websocket_streaming", "status": "FAIL", "error": str(e)}
    
    def test_concurrent_requests(self) -> Dict[str, Any]:
        """Test concurrent request handling"""
        logger.info("\n=== Testing Concurrent Requests ===")
        
        import concurrent.futures
        
        def make_request(index):
            try:
                response = requests.post(
                    f"{self.server_url}/tts",
                    json={
                        "text": f"This is concurrent request number {index}.",
                        "voice": "tara",
                        "max_tokens": 500
                    },
                    timeout=30
                )
                return response.status_code == 200
            except:
                return False
        
        num_requests = 5
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(make_request, i) for i in range(num_requests)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        total_time = time.time() - start_time
        success_count = sum(results)
        
        result = {
            "test": "concurrent_requests",
            "status": "PASS" if success_count == num_requests else "PARTIAL",
            "requests_sent": num_requests,
            "requests_successful": success_count,
            "total_time_s": total_time,
            "avg_time_per_request": total_time / num_requests
        }
        
        logger.info(f"{'✅' if success_count == num_requests else '⚠️'} Concurrent requests: {success_count}/{num_requests} successful")
        logger.info(f"   Total time: {total_time:.1f}s")
        logger.info(f"   Average per request: {result['avg_time_per_request']:.1f}s")
        
        return result
    
    def run_all_tests(self) -> List[Dict[str, Any]]:
        """Run all tests in the suite"""
        logger.info("\n" + "="*50)
        logger.info("ORPHEUS TTS COMPREHENSIVE TEST SUITE")
        logger.info("="*50)
        
        # Check server health first
        if not self.check_server_health():
            logger.error("❌ Server is not responding! Please start the server first.")
            return []
        
        logger.info("✅ Server is healthy, starting tests...")
        
        # Run all tests
        all_results = []
        
        # Synchronous tests
        all_results.append(self.test_basic_generation())
        all_results.append(self.test_different_voices())
        all_results.append(self.test_emotion_tags())
        all_results.append(self.test_concurrent_requests())
        
        # Test long generation (optional, takes time)
        logger.info("\n=== Long Generation Test (Optional) ===")
        response = input("Run long generation test? This may take several minutes (y/n): ")
        if response.lower() == 'y':
            all_results.append(self.test_long_generation())
        
        # Async tests
        logger.info("\n=== WebSocket Test ===")
        try:
            result = asyncio.run(self.test_websocket_streaming())
            all_results.append(result)
        except Exception as e:
            logger.error(f"WebSocket test failed: {e}")
            all_results.append({"test": "websocket_streaming", "status": "FAIL", "error": str(e)})
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("TEST SUMMARY")
        logger.info("="*50)
        
        passed = sum(1 for r in all_results if r.get("status") == "PASS")
        failed = sum(1 for r in all_results if r.get("status") == "FAIL")
        partial = sum(1 for r in all_results if r.get("status") == "PARTIAL")
        
        logger.info(f"Total tests: {len(all_results)}")
        logger.info(f"✅ Passed: {passed}")
        logger.info(f"❌ Failed: {failed}")
        logger.info(f"⚠️  Partial: {partial}")
        
        # Save results to file
        results_file = self.output_dir / "test_results.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        logger.info(f"\nResults saved to: {results_file}")
        
        return all_results

def run_all_tests(server_url: str = "http://localhost:8080"):
    """Entry point for running all tests"""
    suite = OrpheusTestSuite(server_url)
    return suite.run_all_tests()

if __name__ == "__main__":
    import sys
    
    # Allow custom server URL
    server_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    
    # Run tests
    results = run_all_tests(server_url)
    
    # Exit with appropriate code
    if results:
        failed = sum(1 for r in results if r.get("status") == "FAIL")
        sys.exit(0 if failed == 0 else 1)
    else:
        sys.exit(1)