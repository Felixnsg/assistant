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

# Set PyTorch memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Create the FastAPI app
app = FastAPI(title="Orpheus TTS Server (Memory Optimized)")

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

def check_gpu_memory():
    """Check available GPU memory and kill processes if needed"""
    try:
        import torch
        if not torch.cuda.is_available():
            return "No CUDA GPU available"
        
        # Get current memory usage
        free_mem, total_mem = torch.cuda.mem_get_info(0)
        free_gb = free_mem / (1024**3)
        total_gb = total_mem / (1024**3)
        
        print(f"GPU memory: {free_gb:.2f}GB free / {total_gb:.2f}GB total")
        
        # If less than 5GB free, try to free up memory
        if free_gb < 5:
            print("Low GPU memory. Attempting to free up resources...")
            
            # Clear PyTorch cache
            torch.cuda.empty_cache()
            
            # Check memory again
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            free_gb = free_mem / (1024**3)
            
            print(f"After cleanup: {free_gb:.2f}GB free / {total_gb:.2f}GB total")
            
            if free_gb < 5:
                return f"WARNING: Only {free_gb:.2f}GB free - may encounter memory issues"
        
        return f"GPU memory check: {free_gb:.2f}GB free / {total_gb:.2f}GB total"
    except Exception as e:
        return f"Error checking GPU memory: {str(e)}"

@app.on_event("startup")
async def startup_event():
    global model, model_loading_error
    
    # Check GPU memory
    memory_status = check_gpu_memory()
    print(memory_status)
    
    try:
        # Clear GPU memory before loading the model
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
            
            # Print GPU info
            device = torch.cuda.current_device()
            device_name = torch.cuda.get_device_name(device)
            print(f"Using GPU: {device_name}")
            
            # Set to low precision to save memory
            torch.set_default_dtype(torch.float16)
            print("Set default precision to float16 to save memory")
        
        print("Attempting to load Orpheus TTS model...")
        
        # First try without importing vllm directly to avoid memory leaks
        try:
            from orpheus_tts import OrpheusModel
            model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
            print("Model loaded successfully!")
            return
        except Exception as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory error: {e}")
                print("Trying again with even more aggressive memory cleanup...")
                
                # More aggressive memory cleanup
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                
                # Try to load in smaller chunks
                torch.cuda.set_per_process_memory_fraction(0.8)  # Only use 80% of available memory
                
                model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
                print("Model loaded successfully after memory optimization!")
                return
            else:
                raise
        
    except ImportError as e:
        error_msg = f"Error importing libraries: {str(e)}"
        print(f"ERROR: {error_msg}")
        model_loading_error = error_msg
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        model_loading_error = error_msg

@app.get("/")
async def root():
    memory_status = check_gpu_memory()
    return {
        "message": "Orpheus TTS Server is running!",
        "model_loaded": model is not None,
        "error": model_loading_error if model_loading_error else None,
        "gpu_memory_status": memory_status
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
        # Check GPU memory before processing
        memory_status = check_gpu_memory()
        print(f"Memory before processing: {memory_status}")
        
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
    # Don't use reload=True as it can cause GPU memory issues
    uvicorn.run("memory_optimized:app", host="0.0.0.0", port=8080, reload=False)