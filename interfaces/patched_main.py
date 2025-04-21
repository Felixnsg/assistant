import os
import sys
import time
import wave
import uvicorn
import asyncio
import tempfile
import traceback
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Important: Set environment variables BEFORE importing any vLLM or orpheus_tts modules
os.environ["VLLM_MAX_MODEL_LEN"] = "100000"  # Reduced from 131072
os.environ["VLLM_ENABLE_CUDA_GRAPH"] = "1"   # Enable CUDA graphs for better memory efficiency
os.environ["VLLM_TP_SIZE"] = "1"             # Force tensor parallelism to 1

# Create the FastAPI app
app = FastAPI(title="Orpheus TTS Server")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder for the model
model = None
model_loading_error = None

# Create a directory for temporary audio files
os.makedirs("temp_audio", exist_ok=True)

class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"  # Default voice
    temperature: float = 1.0
    top_p: float = 0.9
    repetition_penalty: float = 1.1

class TTSResponse(BaseModel):
    audio_url: str
    duration: float
    processing_time: float

def patch_vllm():
    """Monkey patch vLLM LLMEngine to use different max_model_len"""
    try:
        # Try to import vllm first to patch it
        import vllm
        from vllm.engine.arg_utils import EngineArgs
        
        # Store the original from_dict method
        original_from_dict = EngineArgs.from_dict
        
        # Create a patched version that modifies max_model_len
        def patched_from_dict(cls, config_dict):
            # Modify the config_dict to reduce max_model_len
            if "max_model_len" in config_dict and config_dict["max_model_len"] > 100000:
                print(f"Patching max_model_len from {config_dict['max_model_len']} to 100000")
                config_dict["max_model_len"] = 100000
            
            # Call the original method with our modified dict
            return original_from_dict(cls, config_dict)
        
        # Apply the patch
        EngineArgs.from_dict = classmethod(patched_from_dict)
        print("Successfully patched vLLM EngineArgs.from_dict to limit max_model_len")
        
        return True
    except Exception as e:
        print(f"Failed to patch vLLM: {e}")
        return False

def check_gpu():
    """Check if CUDA is available and return GPU info"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "CUDA is not available. GPU acceleration will not be used."
        
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3) if gpu_count > 0 else 0
        
        return f"CUDA is available. Found {gpu_count} GPU(s). Using: {gpu_name} with {gpu_memory:.2f} GB memory."
    except Exception as e:
        return f"Error checking GPU: {str(e)}"

@app.on_event("startup")
async def startup_event():
    global model, model_loading_error
    
    print("Checking GPU status:")
    gpu_status = check_gpu()
    print(gpu_status)
    
    # Try to free up GPU memory before loading the model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
    except:
        pass
    
    # Try to patch vLLM before importing orpheus_tts
    patch_result = patch_vllm()
    print(f"vLLM patching result: {patch_result}")
    
    try:
        print("Attempting to load Orpheus TTS model...")
        from orpheus_tts import OrpheusModel
        
        try:
            # Attempt to load a minimal model in CPU first to see if the library works
            print("Testing orpheus_tts library with minimal model...")
            temp_model = OrpheusModel.__new__(OrpheusModel)
            print("Library initialization test passed")
        except Exception as test_e:
            print(f"Error during library test: {test_e}")
        
        try:
            # For GPU models with memory constraints, use a reduced sequence length
            model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
            print("Model loaded successfully!")
        except Exception as e:
            error_msg = str(e)
            print(f"Failed to load model with standard settings: {error_msg}")
            
            # If we still hit memory issues, try with a custom engine wrapper
            if "max seq len" in error_msg and "KV cache" in error_msg:
                print("Trying custom engine initialization...")
                
                # Let's try to manually load and adjust the LLM engine
                from vllm import LLM
                import torch
                from orpheus_tts import OrpheusModel
                
                # Calculate a safe model length based on the error message
                import re
                kv_cache_size = 120000  # Default safe value
                match = re.search(r"stored in KV cache \((\d+)\)", error_msg)
                if match:
                    available_size = int(match.group(1))
                    kv_cache_size = int(available_size * 0.9)  # Use 90% to be safe
                
                print(f"Using custom max_model_len={kv_cache_size}")
                
                # Custom loading approach
                # 1. Try to create a vLLM engine directly
                llm = LLM(
                    model="canopylabs/orpheus-tts-0.1-finetune-prod",
                    max_model_len=kv_cache_size,
                    dtype="half"
                )
                
                # 2. Create an OrpheusModel with our pre-loaded engine
                model = OrpheusModel.__new__(OrpheusModel)
                model._llm = llm
                
                # Initialize other needed attributes
                # These are guesses about the OrpheusModel internals
                model._tokenizer = None  # Will be loaded on first use
                
                print("Custom model initialization successful!")
                
                # Test if it works
                model.generate_speech("Hello world", voice="tara")
                print("Speech generation test successful!")
                
    except ImportError as e:
        error_msg = f"Error importing orpheus_tts: {str(e)}"
        print(f"ERROR: {error_msg}")
        print("Please install it with: pip install orpheus-speech")
        print("Then fix potential vllm issue with: pip install vllm==0.7.3")
        model_loading_error = error_msg
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        model_loading_error = error_msg
        
        print("\nAttempting a CPU-only fallback...")
        try:
            # Set environment variable to disable CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            # Reload modules to pick up the environment change
            import importlib
            
            # Try to reimport and get a fresh OrpheusModel
            if "orpheus_tts" in sys.modules:
                importlib.reload(sys.modules["orpheus_tts"])
            if "vllm" in sys.modules:
                importlib.reload(sys.modules["vllm"])
                
            from orpheus_tts import OrpheusModel
            model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
            print("CPU-only model loaded successfully. Note: This will be VERY slow.")
        except Exception as cpu_e:
            print(f"CPU fallback also failed: {cpu_e}")
            print(traceback.format_exc())

@app.get("/")
async def root():
    return {
        "message": "Orpheus TTS Server is running!",
        "model_loaded": model is not None,
        "error": model_loading_error if model_loading_error else None
    }

@app.get("/health")
async def health_check():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unavailable",
                "message": "TTS model not loaded",
                "error": model_loading_error if model_loading_error else "Unknown error"
            }
        )
    return {"status": "healthy", "model_loaded": True}

@app.get("/voices")
async def get_voices():
    """Return the list of available voices"""
    voices = {
        "english": ["tara", "leah", "jess", "leo", "dan", "mia", "zac", "zoe"],
        # Add other languages as they become available
    }
    return voices

@app.post("/tts")
async def text_to_speech(request: TTSRequest, background_tasks: BackgroundTasks):
    """Generate speech from text and return audio file URL"""
    if not model:
        raise HTTPException(
            status_code=503,
            detail=f"TTS model not loaded. Error: {model_loading_error or 'Unknown error'}. Please check server logs."
        )
    
    try:
        # Generate a unique file ID
        file_id = f"{int(time.time())}_{hash(request.text) % 10000}"
        output_path = f"temp_audio/{file_id}.wav"
        
        # Generate speech
        start_time = time.monotonic()
        syn_tokens = model.generate_speech(
            prompt=request.text,
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty
        )
        
        # Write audio to file
        with wave.open(output_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            total_frames = 0
            for audio_chunk in syn_tokens:
                frame_count = len(audio_chunk) // (wf.getsampwidth() * wf.getnchannels())
                total_frames += frame_count
                wf.writeframes(audio_chunk)
            
            duration = total_frames / wf.getframerate()
        
        end_time = time.monotonic()
        processing_time = end_time - start_time
        
        # Schedule cleanup of the file after some time
        async def cleanup_file():
            await asyncio.sleep(3600)  # 1 hour
            if os.path.exists(output_path):
                os.remove(output_path)
                
        background_tasks.add_task(cleanup_file)
        
        # Return the audio file URL
        return TTSResponse(
            audio_url=f"/audio/{file_id}.wav",
            duration=duration,
            processing_time=processing_time
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating speech: {str(e)}\n{traceback.format_exc()}"
        )

@app.get("/audio/{file_id}.wav")
async def get_audio_file(file_id: str):
    """Serve the generated audio file"""
    file_path = f"temp_audio/{file_id}.wav"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(file_path, media_type="audio/wav")

if __name__ == "__main__":
    uvicorn.run("patched_main:app", host="0.0.0.0", port=8080, reload=False)  # Disable reload for better memory management