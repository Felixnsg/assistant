# File: GPUserver.py
# --- REFACTOR: Added file description ---
"""
WebSocket server for GPU-accelerated person detection and Felix recognition.

Receives video frames from a client (IseeYou.py), processes them using
PersonDetector (YOLOv8) and FelixRecognizer, and sends results back.
"""

import asyncio
import websockets
import torch
import cv2
import numpy as np
import json
from PIL import Image # Not strictly needed if processing directly with cv2/numpy
import io # Not used
import traceback
import time
import sys
import os
import logging # --- REFACTOR: Added logging ---

# --- REFACTOR: Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')

# Check if running in interactive mode (less critical now with logging)
# is_interactive = hasattr(sys, 'ps1')
# logging.info(f"Running in interactive mode: {is_interactive}")

# --- REFACTOR: Improved module import and path handling ---
try:
    # Assuming person_detector.py and felix_recognizer.py are in the same dir or Python path
    from person_detector import PersonDetector
    from felix_recognizer import FelixRecognizer
    logging.info("Modules PersonDetector and FelixRecognizer imported successfully.")
except ImportError as e:
    logging.error(f"ERROR importing detector/recognizer modules: {e}", exc_info=True)
    # Try adding parent directory if modules are in ../core or similar structure
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    try:
        # Example: Adjust path if they are in core
        # from core.person_detector import PersonDetector
        # from core.felix_recognizer import FelixRecognizer
        from person_detector import PersonDetector # Retry with potentially updated path
        from felix_recognizer import FelixRecognizer
        logging.info("Modules re-imported successfully after path adjustment.")
    except ImportError as e2:
         logging.error(f"ERROR importing detector/recognizer modules even after path adjustment: {e2}", exc_info=True)
         sys.exit(1) # Exit if critical modules can't be loaded

# --- REFACTOR: Import config ---
try:
    # Assuming config.py is in the parent directory relative to this file's location
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import config
except ImportError as e:
     logging.error(f"Error importing config module: {e}. Ensure config.py exists.", exc_info=True)
     # Use defaults if config import fails, or exit
     class config: # Dummy class with defaults
        FELIX_MODEL_PATH = "/root/models/felix_classifier.pth" # Example default
        YOLO_MODEL_PATH = None # Use default yolo
        PERSON_DETECTION_THRESHOLD = 0.6
        FELIX_RECOGNIZER_THRESHOLD = 0.6
        # Add other needed defaults
     logging.warning("Using default configuration values as config.py could not be imported.")
     # Or sys.exit(1)


class FelixDetectionServer:
    """
    Handles WebSocket connections, receives frames, performs detection/recognition,
    and sends results back to the client.

    Attributes:
        detector (PersonDetector): Instance for detecting persons.
        recognizer (FelixRecognizer): Instance for recognizing Felix.
        frame_count (int): Counter for processed frames in the current session.
        total_detections (int): Counter for total person detections.
        felix_detections (int): Counter for total Felix detections.
    """

    # --- REFACTOR: Improved docstring, use config for model paths/thresholds ---
    def __init__(self,
                 felix_model_path: str = config.FELIX_MODEL_PATH,
                 yolo_model_path: str = config.YOLO_MODEL_PATH,
                 person_threshold: float = config.PERSON_DETECTION_THRESHOLD,
                 felix_threshold: float = config.FELIX_RECOGNIZER_THRESHOLD):
        """
        Initializes the FelixDetectionServer.

        Args:
            felix_model_path (str): Path to the Felix recognizer model weights.
            yolo_model_path (str): Path to the YOLO person detector model (or None for default).
            person_threshold (float): Confidence threshold for person detection.
            felix_threshold (float): Confidence threshold for Felix recognition.

        Raises:
            RuntimeError: If PersonDetector or FelixRecognizer fails to initialize.
        """
        logging.info("Initializing FelixDetectionServer...")

        try:
            # Initialize YOLOv8 person detector
            logging.info("Creating PersonDetector...")
            self.detector = PersonDetector(model_path=yolo_model_path, confidence_threshold=person_threshold)
            logging.info("PersonDetector created.")

            # Initialize Felix recognizer
            logging.info("Creating FelixRecognizer...")
            self.recognizer = FelixRecognizer(model_path=felix_model_path, confidence_threshold=felix_threshold)
            logging.info("FelixRecognizer created.")

            # Stats tracking
            self.frame_count = 0
            self.total_detections = 0
            self.felix_detections = 0

            # Print GPU info
            if torch.cuda.is_available():
                logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
                # logging.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB") # Can be verbose
            else:
                logging.warning("CUDA not available. Using CPU (performance may be significantly reduced).")

            logging.info("FelixDetectionServer initialized successfully.")
        except (RuntimeError, FileNotFoundError) as e:
            logging.error(f"ERROR initializing server component: {e}", exc_info=True)
            raise # Re-raise critical initialization errors
        except Exception as e:
            logging.error(f"Unexpected ERROR initializing server: {e}", exc_info=True)
            raise # Re-raise critical initialization errors

    # --- REFACTOR: Improved docstring, logging, error handling ---
    async def process_frame(self, frame: np.ndarray) -> list[dict]:
        """
        Processes a single video frame to detect persons and recognize Felix.

        Args:
            frame (np.ndarray): The input video frame (OpenCV BGR format).

        Returns:
            list[dict]: A list of dictionaries, each representing a detected person
                        with their bounding box, Felix status, and confidence.
                        Example: [{"box": [x, y, w, h], "is_felix": bool, "confidence": float}]
                        Returns empty list on error.
        """
        self.frame_count += 1
        start_time = time.monotonic()
        # logging.info(f"----- Processing Frame #{self.frame_count} -----") # Can be verbose

        if not isinstance(frame, np.ndarray):
             logging.error("process_frame: Invalid input - frame is not a NumPy array.")
             return []

        results = []
        try:
            # 1. Run person detection
            # logging.debug("Running person detection...") # Use debug level
            person_boxes = self.detector.detect(frame)
            detection_time = time.monotonic()
            # logging.debug(f"Detection found {len(person_boxes)} people in {(detection_time - start_time)*1000:.1f} ms")

            # 2. Run Felix recognition for each detected person
            recognition_tasks = []
            felix_found_in_frame = False
            for i, box_with_conf in enumerate(person_boxes):
                 person_box = box_with_conf[:4] # Get [x, y, w, h]
                 # logging.debug(f"Running recognition for person #{i+1} @ {person_box}...")
                 # --- REFACTOR: Run recognition (consider async if becomes bottleneck, but likely sequential is fine) ---
                 try:
                      is_felix, confidence = self.recognizer.is_felix(frame, person_box)
                      result_type = "FELIX" if is_felix else "NOT FELIX"
                      # logging.debug(f"Person #{i+1} is {result_type} (confidence: {confidence:.3f})")
                      if is_felix:
                          felix_found_in_frame = True

                      results.append({
                          "box": [int(coord) for coord in person_box], # Ensure integer coords in result
                          "is_felix": bool(is_felix),
                          "confidence": float(confidence) # Ensure float
                      })
                 except Exception as recog_e:
                      logging.error(f"Error during Felix recognition for box {person_box}: {recog_e}", exc_info=True)
                      # Optionally add a placeholder result indicating error for this box
                      # results.append({"box": person_box, "is_felix": False, "confidence": 0.0, "error": str(recog_e)})


            # Update counters
            self.total_detections += len(person_boxes)
            if felix_found_in_frame:
                 self.felix_detections += 1 # Count frames where Felix is detected at least once

            processing_time = time.monotonic() - start_time
            # Log only if detections found or processing is slow
            if results or processing_time > 0.5:
                 logging.info(f"Frame #{self.frame_count}: Found {len(results)} people ({sum(1 for r in results if r['is_felix'])} Felix). Time: {processing_time*1000:.1f} ms")

            return results

        except Exception as e:
            logging.error(f"Error in process_frame: {e}", exc_info=True)
            return [] # Return empty list on error

    # --- REFACTOR: Improved docstring, logging, error handling ---
    async def handle_client(self, websocket, path: str):
        """
        Handles a single client WebSocket connection.

        Receives frames, processes them, and sends back results.

        Args:
            websocket (websockets.WebSocketServerProtocol): The WebSocket connection object.
            path (str): The connection path (not used here).
        """
        client_addr = websocket.remote_address
        logging.info(f"+++ Client connected: {client_addr} +++")
        try:
            async for message in websocket:
                frame_start_time = time.monotonic()
                try:
                    # Check message type (expecting bytes)
                    if not isinstance(message, bytes):
                         logging.warning(f"Received non-bytes message from {client_addr}. Skipping.")
                         continue

                    # Decode the frame from binary data (assuming JPEG format from client)
                    frame_bytes = np.frombuffer(message, dtype=np.uint8)
                    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)

                    if frame is None:
                        logging.error(f"Could not decode frame received from {client_addr}.")
                        # Optionally send error back to client
                        # await websocket.send(json.dumps({"error": "Failed to decode frame"}))
                        continue # Skip this message

                    # Process the frame
                    results = await self.process_frame(frame)

                    # Send back the results as JSON
                    response_payload = json.dumps(results)
                    await websocket.send(response_payload)

                    # Log transfer/processing time less frequently
                    # if self.frame_count % 30 == 0:
                    #      logging.info(f"Sent results for frame #{self.frame_count} to {client_addr} ({len(results)} detections)")

                # --- REFACTOR: Catch specific websocket errors ---
                except websockets.exceptions.ConnectionClosed:
                     logging.warning(f"Connection closed unexpectedly by client {client_addr}.")
                     break # Exit loop on connection closed
                except json.JSONDecodeError as e:
                     logging.error(f"Error encoding results to JSON: {e}", exc_info=True)
                     # Don't send potentially broken data
                except cv2.error as e:
                     logging.error(f"OpenCV error processing frame from {client_addr}: {e}", exc_info=True)
                except Exception as e:
                    logging.error(f"Error processing message from {client_addr}: {e}", exc_info=True)
                    # Optionally send generic error back to client if connection is open
                    try:
                        await websocket.send(json.dumps({"error": "Internal server error during processing"}))
                    except websockets.exceptions.ConnectionClosed:
                         pass # Ignore if connection closed while trying to send error
                    except Exception as send_e:
                         logging.error(f"Error sending error message to client {client_addr}: {send_e}")

        except websockets.exceptions.ConnectionClosedOK:
            logging.info(f"Client {client_addr} disconnected normally.")
        except websockets.exceptions.ConnectionClosedError as e:
             logging.warning(f"Client {client_addr} connection closed with error: {e}")
        except Exception as e:
            logging.error(f"Error handling client {client_addr}: {e}", exc_info=True)
        finally:
            logging.info(f"--- Client disconnected: {client_addr} ---")

# --- REFACTOR: Improved main function with error handling ---
async def main():
    """Sets up and runs the FelixDetectionServer."""
    logging.info("Starting main function...")

    # --- REFACTOR: Configuration loading handled by importing config ---
    # Model paths and thresholds are now read from config object

    try:
        # Create the server instance
        logging.info("Creating FelixDetectionServer instance...")
        server = FelixDetectionServer(
            felix_model_path=config.FELIX_MODEL_PATH,
            yolo_model_path=config.YOLO_MODEL_PATH,
            person_threshold=config.PERSON_DETECTION_THRESHOLD,
            felix_threshold=config.FELIX_RECOGNIZER_THRESHOLD
        )
        logging.info("Server instance created successfully.")

        # Configure WebSocket server parameters
        host = "0.0.0.0"
        port = 8765
        # --- REFACTOR: Use configuration for limits if needed ---
        max_size = 10 * 1024 * 1024  # 10MB message size limit
        ping_interval = 20
        ping_timeout = 20

        logging.info(f"Starting WebSocket server on ws://{host}:{port}")
        async with websockets.serve(
            lambda websocket, path: server.handle_client(websocket, path),
            host,
            port,
            ping_interval=ping_interval,
            ping_timeout=ping_timeout,
            max_size=max_size
        ):
       
            logging.info(f"=== Server running ===")
            logging.info(f"=== Max message size: {max_size / (1024*1024):.1f} MB ===")
            logging.info(f"=== Waiting for client connections ===")
            await asyncio.Future()  # Run forever until interrupted

    except (RuntimeError, FileNotFoundError) as e:
         logging.error(f"FATAL: Failed to initialize server components: {e}", exc_info=True)
    except OSError as e:
         # Catch common network errors like "address already in use"
         logging.error(f"FATAL: Could not start WebSocket server (OS Error): {e}", exc_info=True)
    except Exception as e:
        logging.error(f"FATAL ERROR in main server setup: {e}", exc_info=True)

if __name__ == "__main__":
    try:
        logging.info("Starting Felix Detection Server...")
        asyncio.run(main())
    except KeyboardInterrupt:
        logging.info("\nServer stopped by user (KeyboardInterrupt).")
    except Exception as e:
        # Catch any unexpected errors during asyncio.run or shutdown
        logging.critical(f"FATAL ERROR: Unhandled exception during server execution: {e}", exc_info=True)