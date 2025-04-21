import os
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

# Create the FastAPI app
app = FastAPI(title="Orpheus TTS vLLM Server")

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
tokenizer = None

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
    global model, model_loading_error, tokenizer
    
    # Try to free up GPU memory before loading the model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("GPU cache cleared")
    except:
        pass
    
    try:
        # First, try loading the standard orpheus_tts model
        try:
            print("Attempting to load Orpheus TTS model...")
            from orpheus_tts import OrpheusModel
            
            model = OrpheusModel(model_name="canopylabs/orpheus-tts-0.1-finetune-prod")
            print("Model loaded successfully!")
            return
        except Exception as e:
            error_msg = str(e)
            print(f"Failed to load using standard method: {error_msg}")
            
            # If there's a KV cache issue, let's try a more direct approach using vLLM
            if "max seq len" in error_msg and "KV cache" in error_msg:
                print("Attempting to load using direct vLLM approach...")
                
                # Extract KV cache limit if possible
                import re
                kv_cache_limit = 80000  # Default conservative value
                match = re.search(r"stored in KV cache \((\d+)\)", error_msg)
                if match:
                    kv_cache_limit = int(match.group(1)) - 5000  # Keep a small buffer
                
                try:
                    from vllm import LLM, SamplingParams
                    from transformers import AutoTokenizer
                    
                    # Load tokenizer
                    print("Loading tokenizer...")
                    tokenizer = AutoTokenizer.from_pretrained("canopylabs/orpheus-tts-0.1-finetune-prod")
                    
                    # Load the model with custom parameters
                    print(f"Loading LLM with max_model_len={kv_cache_limit}...")
                    model = LLM(
                        model="canopylabs/orpheus-tts-0.1-finetune-prod",
                        max_model_len=kv_cache_limit,
                        gpu_memory_utilization=0.95,
                        dtype="half"
                    )
                    
                    print("Model loaded successfully via direct vLLM approach!")
                    
                    # Simple test to verify it works
                    test_input = f"tara: Hello, this is a test."
                    test_output = model.generate(
                        [test_input],
                        SamplingParams(
                            temperature=1.0,
                            top_p=0.9,
                            repetition_penalty=1.1,
                            max_tokens=100
                        )
                    )
                    
                    print("Model test run successful!")
                    
                    # Define a custom generate_speech function to mimic OrpheusModel's interface
                    def custom_generate_speech(prompt, voice="tara", temperature=1.0, top_p=0.9, repetition_penalty=1.1):
                        # Format the prompt properly for the model
                        formatted_prompt = f"{voice}: {prompt}"
                        
                        # Generate text tokens
                        outputs = model.generate(
                            [formatted_prompt],
                            SamplingParams(
                                temperature=temperature,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                max_tokens=2000  # Adjust based on prompt length
                            )
                        )
                        
                        output_text = outputs[0].outputs[0].text
                        
                        # Convert the output tokens to audio
                        # Note: We'd need to implement the tokenizer → audio conversion
                        # This is where we'd use the Orpheus speech synthesis part
                        
                        # For now, just return a placeholder
                        print("ERROR: Custom speech generation not fully implemented!")
                        raise NotImplementedError(
                            "Custom speech generation via direct vLLM is not implemented. "
                            "Please use the standard OrpheusModel method."
                        )
                    
                    # Attach the method to our model object
                    model.generate_speech = custom_generate_speech
                    
                except Exception as vllm_err:
                    print(f"Failed to load using vLLM approach: {str(vllm_err)}")
                    raise
    
    except ImportError as e:
        error_msg = f"Error importing required libraries: {str(e)}"
        print(f"ERROR: {error_msg}")
        print("Please install required packages: pip install orpheus-speech vllm transformers")
        model_loading_error = error_msg
    except Exception as e:
        error_msg = f"Error loading model: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(f"Full traceback: {traceback.format_exc()}")
        model_loading_error = error_msg

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
    uvicorn.run("vllm_server:app", host="0.0.0.0", port=8080, reload=True)