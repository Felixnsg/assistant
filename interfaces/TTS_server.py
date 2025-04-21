import os
import time
import wave
import uvicorn
import asyncio
import tempfile
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
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Placeholder for the model
model = None

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

@app.on_event("startup")
async def startup_event():
    global model
    try:
        from orpheus_tts import OrpheusModel
        print("Loading Orpheus TTS model...")
        model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
        print("Model loaded successfully!")
    except ImportError:
        print("Error: orpheus_tts package not installed")
        print("Please install it with: pip install orpheus-speech")
        print("Then fix potential vllm issue with: pip install vllm==0.7.3")
    except Exception as e:
        print(f"Error loading model: {e}")

@app.get("/")
async def root():
    return {"message": "Orpheus TTS Server is running!", "model_loaded": model is not None}

@app.get("/health")
async def health_check():
    if model is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "message": "TTS model not loaded"}
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
        raise HTTPException(status_code=503, detail="TTS model not loaded. Please check server logs.")
    
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
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

@app.get("/audio/{file_id}.wav")
async def get_audio_file(file_id: str):
    """Serve the generated audio file"""
    file_path = f"temp_audio/{file_id}.wav"
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Audio file not found")
    
    return FileResponse(file_path, media_type="audio/wav")

@app.post("/tts/inline")
async def text_to_speech_inline(request: TTSRequest):
    """Generate speech and return the audio file directly"""
    if not model:
        raise HTTPException(status_code=503, detail="TTS model not loaded. Please check server logs.")
    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = temp_file.name
        
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
        with wave.open(temp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(24000)
            
            for audio_chunk in syn_tokens:
                wf.writeframes(audio_chunk)
        
        # Return the file and then delete it
        return FileResponse(
            temp_path, 
            media_type="audio/wav",
            headers={"X-Processing-Time": str(time.monotonic() - start_time)},
            background=lambda: os.remove(temp_path) if os.path.exists(temp_path) else None
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating speech: {str(e)}")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=True)