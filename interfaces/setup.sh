import os
import time
import wave
import uvicorn
import asyncio
import tempfile
import traceback
import sys
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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

def verify_dependencies():
    """Verify that all required dependencies are installed"""
    missing_deps = []
    
    try:
        import torch
    except ImportError:
        missing_deps.append("torch")
    
    try:
        import vllm
    except ImportError:
        missing_deps.append("vllm")
    
    try:
        import orpheus_tts
    except ImportError:
        missing_deps.append("orpheus-speech")
    
    return missing_deps

@app.on_event("startup")
async def startup_event():
    global model, model_loading_error
    
    print("Checking GPU status:")
    gpu_status = check_gpu()
    print(gpu_status)
    
    # Check dependencies
    missing_deps = verify_dependencies()
    if missing_deps:
        error_msg = f"Missing required dependencies: {', '.join(missing_deps)}"
        print(f"ERROR: {error_msg}")
        model_loading_error = error_msg
        return
    
    try:
        from orpheus_tts import OrpheusModel
        print("Attempting to load Orpheus TTS model...")
        
        # Try to load with explicit device specification
        try:
            print("Loading model with device='cuda:0'...")
            model = OrpheusModel(
                model_name="canopylabs/orpheus-tts-0.1-finetune-prod",
                device="cuda:0"
            )
            print("Model loaded successfully with device specification!")
            return
        except Exception as e:
            print(f"Failed to load model with device specification: {str(e)}")
            print("Trying alternative loading method...")
        
        # Fall back to default loading
        model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        print("Model loaded successfully with default settings!")
        
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
        
        # Try CPU fallback if GPU failed
        try:
            print("Attempting to load model in CPU-only mode (will be very slow)...")
            from orpheus_tts import OrpheusModel
            
            # Set environment variable to disable CUDA
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            
            model = OrpheusModel(
                model_name="canopylabs/orpheus-tts-0.1-finetune-prod"
            )
            print("Model loaded successfully in CPU-only mode!")
        except Exception as cpu_e:
            print(f"CPU fallback also failed: {str(cpu_e)}")

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

@app.get("/debug")
async def debug_info():
    """Return debugging information"""
    import sys
    import platform
    
    gpu_info = check_gpu()
    
    python_info = {
        "version": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable
    }
    
    # Get package versions
    packages = {}
    try:
        import pkg_resources
        for pkg in ["torch", "vllm", "orpheus_tts", "fastapi", "uvicorn"]:
            try:
                packages[pkg] = pkg_resources.get_distribution(pkg).version
            except pkg_resources.DistributionNotFound:
                packages[pkg] = "Not installed"
    except ImportError:
        packages = "Unable to get package information"
    
    return {
        "model_loaded": model is not None,
        "model_error": model_loading_error,
        "gpu_info": gpu_info,
        "python_info": python_info,
        "packages": packages,
        "environment_variables": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "Not set")
        }
    }

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
    uvicorn.run("improved_main:app", host="0.0.0.0", port=8080, reload=True)