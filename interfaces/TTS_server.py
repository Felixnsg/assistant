import os
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch

# Assume Orpheus is installed in the venv
from orpheus_tts import OrpheusModel

# --- Configuration ---
MODEL_NAME = "canopylabs/orpheus-tts-0.1-finetune-prod"
HOST = "0.0.0.0"  # Listen on all available network interfaces
PORT = 8080
LOG_LEVEL = logging.INFO

# Expected audio properties based on Orpheus examples/docs (Verify if possible)
SAMPLE_RATE = 24000
NUM_CHANNELS = 1
SAMPLE_WIDTH = 2  # Bytes per sample (16-bit)

# --- Logging Setup ---
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Global State ---
# Dictionary to hold the model instance
model_state = {}

# --- FastAPI Lifespan Manager (Loads model on startup) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the model during startup
    logger.info(f"Loading Orpheus model: {MODEL_NAME}...")
    start_time = time.time()
    try:
        # Check GPU availability
        if not torch.cuda.is_available():
             logger.error("CUDA (GPU) not available. Orpheus TTS requires a GPU.")
             # Decide if you want to raise an error or try CPU (likely too slow)
             # raise RuntimeError("CUDA not available")
             # For now, we'll let it try loading, it might fail gracefully or use CPU
        else:
             logger.info(f"CUDA available. Device: {torch.cuda.get_device_name(0)}")

        model_state["orpheus_model"] = OrpheusModel(model_name=MODEL_NAME)
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds.")
    except Exception as e:
        logger.exception(f"Failed to load Orpheus model: {e}")
        # Optionally exit or prevent server start if model load fails
        # raise RuntimeError("Failed to load model") from e
        model_state["orpheus_model"] = None # Indicate failure

    yield  # Server runs here

    # Clean up resources (optional, good practice)
    logger.info("Shutting down. Clearing model state.")
    model_state.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# --- FastAPI App ---
app = FastAPI(lifespan=lifespan)

# --- Request Body Model ---
class SynthesizeRequest(BaseModel):
    text: str
    voice: str = "tara"  # Default voice
    temperature: float = 0.7 # Optional LLM params
    top_p: float = 0.95    # Optional LLM params
    repetition_penalty: float = 1.2 # Required >= 1.1

# --- Audio Streaming Generator ---
# Note: Orpheus generate_speech already returns an iterator/generator
# So we can often use it directly with StreamingResponse.
# Creating a separate async generator can be useful for adding logs/logic per chunk.

async def audio_stream_generator(model: OrpheusModel, request: SynthesizeRequest):
    """Async generator wrapper for Orpheus synthesis."""
    logger.info(f"Generating speech for voice '{request.voice}' with text: '{request.text[:50]}...'")
    synthesis_start_time = time.monotonic()
    chunk_counter = 0
    total_bytes = 0

    try:
        # Orpheus generate_speech returns an iterator of byte chunks
        # Pass generation parameters directly
        token_iterator = model.generate_speech(
            prompt=request.text, # Orpheus package handles formatting like "tara: text..."
            voice=request.voice,
            temperature=request.temperature,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )

        # Yield chunks as they become available
        for audio_chunk in token_iterator:
            if audio_chunk: # Ensure chunk is not empty
                chunk_counter += 1
                total_bytes += len(audio_chunk)
                # logger.debug(f"Yielding chunk {chunk_counter}, size: {len(audio_chunk)} bytes")
                yield audio_chunk
            # Optional: Add a small sleep if needed to prevent overwhelming client/network,
            # but usually not necessary with proper client handling.
            # await asyncio.sleep(0.001)

        synthesis_duration = time.monotonic() - synthesis_start_time
        audio_duration_sec = total_bytes / (SAMPLE_RATE * NUM_CHANNELS * SAMPLE_WIDTH)
        logger.info(f"Finished streaming {chunk_counter} chunks, {total_bytes} bytes ({audio_duration_sec:.2f}s audio). Generation took {synthesis_duration:.2f}s.")

    except Exception as e:
        logger.exception(f"Error during speech generation: {e}")
        # Optionally yield an error indicator or just stop
        # yield b"ERROR" # Client would need to handle this

# --- API Endpoint ---
@app.post("/synthesize")
async def synthesize_endpoint(request: SynthesizeRequest):
    """
    Accepts text and voice, streams back synthesized audio.
    """
    if "orpheus_model" not in model_state or model_state["orpheus_model"] is None:
        raise HTTPException(status_code=503, detail="TTS Model is not available or failed to load.")

    model = model_state["orpheus_model"]

    # Use the specific media type indicating raw PCM data properties
    # This helps the client interpret the stream correctly
    media_type = f"audio/l16; rate={SAMPLE_RATE}; channels={NUM_CHANNELS}"
    # Alternatively, use 'audio/octet-stream' if client handles format inference

    return StreamingResponse(
        audio_stream_generator(model, request),
        media_type=media_type
    )

@app.get("/")
async def root():
    return {"message": "Orpheus TTS Server is running."}

# --- Run the server (using uvicorn command line is often preferred) ---
# You would typically run this script using:
# uvicorn server:app --host 0.0.0.0 --port 8080 --reload (for development)
# or just:
# uvicorn server:app --host 0.0.0.0 --port 8080 (for production)

if __name__ == "__main__":
    # This block allows running `python server.py` directly for simple testing
    # Production deployments should use a process manager with the uvicorn command above
    import uvicorn
    logger.info("Starting Uvicorn server directly...")
    uvicorn.run(app, host=HOST, port=PORT, log_level=logging.getLevelName(LOG_LEVEL).lower())