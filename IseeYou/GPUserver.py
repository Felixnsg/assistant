# GPUserver.py

import asyncio
import websockets
import torch
import cv2
import numpy as np
import json
from PIL import Image
import io
from person_detector import PersonDetector  # Assuming these are in the same directory or PYTHONPATH
from felix_recognizer import FelixRecognizer # Assuming these are in the same directory or PYTHONPATH
import traceback
import time
import sys
import os
import logging # Use logging instead of just print for better practice

# --- Basic Logging Setup ---
# (Ideally, use a more robust setup like in your main.py)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger("FelixDetectionServer")
# ---------------------------


class FelixDetectionServer:
    """Handles GPU-accelerated detection and recognition (Non-Blocking)"""

    def __init__(self, felix_model, yolo_model=None):
        logger.info("Initializing FelixDetectionServer...")
        # Device selection should happen here if needed by models immediately
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            logger.info("Creating person detector...")
            # Pass device if detector/recognizer need it during init
            self.detector = PersonDetector(model_path=yolo_model)
            logger.info("Person detector created")

            logger.info("Creating Felix recognizer...")
            self.recognizer = FelixRecognizer(felix_model) # Recognizer likely handles device internally
            logger.info("Felix recognizer created")

            self.frame_count = 0 # Instance variable for frame count
            self.total_detections = 0
            self.felix_detections = 0

            if torch.cuda.is_available():
                 logger.info(f"GPU Found: {torch.cuda.get_device_name(0)}")
                 logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                 logger.info("CUDA not available. Using CPU.")

            logger.info("FelixDetectionServer initialized successfully")
        except Exception as e:
            logger.critical(f"ERROR initializing server components: {e}", exc_info=True)
            raise # Reraise after logging


    # --- Make process_frame async and use to_thread ---
    async def process_frame(self, frame):
        """Process a frame using non-blocking calls and return detection results"""
        processing_start_time = time.time()
        self.frame_count += 1 # Assuming this should still be instance level
        current_frame_num = self.frame_count # Capture for logging this frame
        logger.info(f"----- Processing Frame #{current_frame_num} -----")

        results = []
        try:
            # --- Run person detection in thread ---
            detect_start_time = time.time()
            logger.debug(f"Frame #{current_frame_num}: Submitting detection task...")
            try:
                # Offload the blocking detector.detect call
                person_boxes = await asyncio.to_thread(self.detector.detect, frame)
            except Exception as detect_err:
                 logger.error(f"Frame #{current_frame_num}: Error during detection: {detect_err}", exc_info=True)
                 person_boxes = [] # Continue with empty detections on error

            detect_duration = time.time() - detect_start_time
            logger.info(f"Frame #{current_frame_num}: Detection completed in {detect_duration:.4f}s - Found {len(person_boxes)} people")

            if not person_boxes:
                 processing_duration = time.time() - processing_start_time
                 logger.info(f"----- Finished Frame #{current_frame_num} in {processing_duration:.4f}s (No Detections) -----")
                 return [] # Return early if no people detected

            self.total_detections += len(person_boxes)

            # --- Run recognition for each person in thread ---
            recognition_tasks = []
            for i, box_data in enumerate(person_boxes):
                 # Ensure box_data structure is compatible with recognizer
                 # Assuming box_data is [x, y, w, h, confidence] from detector
                 person_box_coords = box_data[:4] # Extract [x, y, w, h]

                 logger.debug(f"Frame #{current_frame_num}: Submitting recognition task for person #{i+1}...")
                 # Create a task to run recognition in a thread for this person
                 task = asyncio.create_task(
                     asyncio.to_thread(self.recognizer.is_felix, frame, person_box_coords),
                     name=f"Recognize_Frame{current_frame_num}_Person{i+1}"
                 )
                 recognition_tasks.append((task, person_box_coords)) # Store task and box coords


            # --- Gather recognition results concurrently ---
            recog_start_time = time.time()
            recognition_results_raw = await asyncio.gather(*(task for task, _ in recognition_tasks), return_exceptions=True)
            recog_duration = time.time() - recog_start_time
            logger.debug(f"Frame #{current_frame_num}: Recognition tasks gathered in {recog_duration:.4f}s")


            # --- Process recognition results ---
            for i, (recog_result, person_box) in enumerate(zip(recognition_results_raw, (box for _, box in recognition_tasks))):

                if isinstance(recog_result, Exception):
                    logger.error(f"Frame #{current_frame_num}: Recognition failed for person #{i+1}: {recog_result}", exc_info=recog_result)
                    # Optionally add a placeholder or skip this detection
                    continue # Skip this person if recognition failed

                # Unpack result from is_felix (assuming it returns (is_felix_bool, confidence_float))
                is_felix, confidence = recog_result
                result_type = "FELIX" if is_felix else "NOT FELIX"
                logger.info(f"Frame #{current_frame_num}: Person #{i+1} is {result_type} (confidence: {confidence:.3f})")

                if is_felix:
                    self.felix_detections += 1

                results.append({
                    "box": [int(c) for c in person_box], # Ensure box coords are int
                    "is_felix": bool(is_felix),
                    "confidence": float(confidence)
                })

            processing_duration = time.time() - processing_start_time
            logger.info(f"----- Finished Frame #{current_frame_num} in {processing_duration:.4f}s ({len(results)} valid results) -----")
            return results

        except Exception as e:
            logger.error(f"Frame #{current_frame_num}: Unexpected error in process_frame: {e}", exc_info=True)
            return [] # Return empty list on unexpected error

    # --- Modify handle_client to use to_thread for decode ---
    async def handle_client(self, websocket):
        """Handle a client connection (Non-Blocking)"""
        client_id = websocket.remote_address # Get unique ID for client
        logger.info(f"+++ Client connected: {client_id} +++")
        message_count = 0
        try:
            async for message in websocket:
                message_count += 1
                message_start_time = time.time()
                logger.debug(f"Client {client_id}: Received message #{message_count} (size: {len(message)} bytes)")
                try:
                    # --- Decode the frame in thread ---
                    decode_start_time = time.time()
                    logger.debug(f"Client {client_id}, Msg #{message_count}: Submitting decode task...")
                    try:
                         # Offload the blocking imdecode
                         frame_bytes = np.frombuffer(message, dtype=np.uint8) # frombuffer is fast
                         frame = await asyncio.to_thread(cv2.imdecode, frame_bytes, cv2.IMREAD_COLOR)
                         # ^^^ Allows event loop to run during decode ^^^
                    except Exception as decode_err:
                         logger.error(f"Client {client_id}, Msg #{message_count}: Error during decode: {decode_err}", exc_info=True)
                         frame = None

                    decode_duration = time.time() - decode_start_time
                    logger.debug(f"Client {client_id}, Msg #{message_count}: Decode finished in {decode_duration:.4f}s")

                    if frame is None:
                        logger.error(f"Client {client_id}, Msg #{message_count}: Could not decode frame")
                        # Maybe send specific error back? For now, send empty list.
                        await websocket.send(json.dumps([]))
                        continue

                    # --- Process the frame (now uses async process_frame) ---
                    results = await self.process_frame(frame)

                    # --- Send back the results ---
                    send_start_time = time.time()
                    try:
                         response_json = json.dumps(results)
                    except TypeError as json_err:
                         logger.error(f"Client {client_id}, Msg #{message_count}: Failed to serialize results to JSON: {json_err}", exc_info=True)
                         response_json = json.dumps([]) # Send empty list on serialization error

                    await websocket.send(response_json)
                    send_duration = time.time() - send_start_time
                    message_duration = time.time() - message_start_time
                    logger.debug(f"Client {client_id}, Msg #{message_count}: Sent results (send took {send_duration:.4f}s, total msg time {message_duration:.4f}s)")

                except websockets.exceptions.ConnectionClosed:
                     logger.warning(f"Client {client_id}: Connection closed during message processing.")
                     break # Exit loop if connection closed
                except Exception as e:
                    logger.error(f"Client {client_id}, Msg #{message_count}: Error processing message: {e}", exc_info=True)
                    try:
                        # Attempt to send empty list even on error
                        await websocket.send(json.dumps([]))
                    except websockets.exceptions.ConnectionClosed:
                         logger.warning(f"Client {client_id}: Connection closed while trying to send error response.")
                         break
                    except Exception as send_err:
                         logger.error(f"Client {client_id}: Failed to send error response: {send_err}")


        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Client {client_id}: Disconnected normally.")
        except websockets.exceptions.ConnectionClosedError as e:
             logger.warning(f"Client {client_id}: Connection closed with error: {e}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
        finally:
            logger.info(f"--- Client disconnected: {client_id} ---")
            # Perform any other cleanup for this client if necessary

async def main():
    logger.info("Starting main function...")

    model_path = "/root/models/felix_classifier.pth" # Make this configurable
    yolo_path = None # Make this configurable

    # Basic check for model file (improve this with proper config loading)
    if not os.path.exists(model_path):
        logger.critical(f"FATAL: Felix model file not found at {model_path}")
        # Add logic to search alternative paths if needed
        return

    try:
        logger.info("Creating server instance...")
        server = FelixDetectionServer(
            felix_model=model_path,
            yolo_model=yolo_path
        )
        logger.info("Server instance created successfully")

        # Add ping/timeout settings matching client expectations
        async with websockets.serve(
            server.handle_client,
            "0.0.0.0",
            8080, # Make port configurable
            ping_interval=15, # Example: Check every 15s
            ping_timeout=30,  # Example: Allow 30s for response
            max_size=10*1024*1024 # Keep reasonable size limit
        ):
            logger.info("=== Server running on ws://0.0.0.0:8080 ===")
            await asyncio.Future() # Run forever

    except Exception as e:
        logger.critical(f"ERROR in main server setup or run: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user (KeyboardInterrupt)")
    except Exception as e:
        logger.critical(f"Unhandled exception in top-level: {e}", exc_info=True)