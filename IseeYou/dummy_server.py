import asyncio
import websockets
import torch
import cv2
import numpy as np
import json
import supervision as sv # Import supervision for drawing
from PIL import Image
import io
from person_detector import PersonDetector
from felix_recognizer import FelixRecognizer
import traceback
import time
import sys
import os
import logging

logging.basicConfig(level=logging.DEBUG, # Lower level to see more detail
                    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger("FelixDetectionServer")


class FelixDetectionServer:
    """
    Handles GPU-accelerated detection/recognition and server-side visualization.
    Sends annotated frames back to the client. (Non-Blocking)
    """

    def __init__(self, felix_model, yolo_model=None):
        logger.info("Initializing FelixDetectionServer...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        try:
            logger.info("Creating person detector...")
            self.detector = PersonDetector(model_path=yolo_model)
            logger.info("Person detector created")

            logger.info("Creating Felix recognizer...")
            self.recognizer = FelixRecognizer(felix_model)
            logger.info("Felix recognizer created")

            # --- Initialize Supervision Annotator ---
            # Using the NEW correct name: BoxAnnotator
            self.box_annotator = sv.BoxAnnotator(
                thickness=2
                # Add text annotator if needed later
            )
            logger.info("Supervision annotator created")

            # Frame count can be per-handler now if desired, or keep instance level
            self.global_frame_count = 0

            if torch.cuda.is_available():
                 logger.info(f"GPU Found: {torch.cuda.get_device_name(0)}")
                 logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                 logger.info("CUDA not available. Using CPU.")

            logger.info("FelixDetectionServer initialized successfully")
        except Exception as e:
            logger.critical(f"ERROR initializing server components: {e}", exc_info=True)
            raise


    # --- NEW: Server-side visualization function ---
    def visualize_server_side(self, frame, detection_results):
        """Draws detection results onto the frame. Runs synchronously."""
        if frame is None or not detection_results:
            return frame # Return original frame if no data

        frame_copy = frame.copy()
        logger.debug(f"Visualizing {len(detection_results)} detections server-side.")

        # Convert results to Supervision Detections format for BoxAnnotator
        boxes = []
        confidences = []
        class_ids = [] # 0 = Felix, 1 = Not Felix
        labels = []

        for det in detection_results:
            x, y, w, h = det["box"]
            conf = det["confidence"]
            is_felix = det["is_felix"]
            class_id = 0 if is_felix else 1

            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(class_id)
            # Create labels directly for BoxAnnotator
            label = f"{'Felix' if is_felix else 'Person'}: {conf:.2f}" # Simpler label
            labels.append(label)

        if not boxes: # Handle case where loop results in no boxes
            return frame_copy

        try:
            sv_detections = sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(confidences),
                class_id=np.array(class_ids)
            )

            # Use the pre-initialized annotator
            frame_copy = self.box_annotator.annotate(
                scene=frame_copy,
                detections=sv_detections,
                labels=labels # Pass labels to the annotator
            )
            logger.debug("Annotation complete.")
        except Exception as viz_err:
             logger.error(f"Error during server-side annotation: {viz_err}", exc_info=True)
             # Return original frame on error
             return frame

        return frame_copy


    # --- process_frame remains async, using to_thread for ML ---
    async def process_frame(self, frame, current_frame_num):
        """Process frame for detections/recognitions (Non-Blocking ML)."""
        processing_start_time = time.time()
        logger.info(f"----- Processing Frame #{current_frame_num} -----")
        results = []
        # ...(Detection logic using asyncio.to_thread for self.detector.detect)...
        # ...(Recognition logic using asyncio.to_thread for self.recognizer.is_felix)...
        # (Your existing non-blocking process_frame logic is mostly fine here)
        # Ensure it returns the 'results' list [{box:[], is_felix:bool, confidence:float}, ...]
        try:
            detect_start_time = time.time()
            logger.debug(f"Frame #{current_frame_num}: Submitting detection task...")
            try:
                person_boxes = await asyncio.to_thread(self.detector.detect, frame)
            except Exception as detect_err:
                 logger.error(f"Frame #{current_frame_num}: Error during detection: {detect_err}", exc_info=True)
                 person_boxes = []

            detect_duration = time.time() - detect_start_time
            logger.info(f"Frame #{current_frame_num}: Detection completed in {detect_duration:.4f}s - Found {len(person_boxes)} people")

            if not person_boxes:
                 processing_duration = time.time() - processing_start_time
                 logger.info(f"----- Finished Frame #{current_frame_num} Proc in {processing_duration:.4f}s (No Detections) -----")
                 return [] # Return empty list

            recognition_tasks = []
            for i, box_data in enumerate(person_boxes):
                 person_box_coords = box_data[:4]
                 task = asyncio.create_task(
                     asyncio.to_thread(self.recognizer.is_felix, frame, person_box_coords),
                     name=f"Recognize_Frame{current_frame_num}_Person{i+1}"
                 )
                 recognition_tasks.append((task, person_box_coords))

            recog_start_time = time.time()
            recognition_results_raw = await asyncio.gather(*(task for task, _ in recognition_tasks), return_exceptions=True)
            recog_duration = time.time() - recog_start_time
            logger.debug(f"Frame #{current_frame_num}: Recognition gathered in {recog_duration:.4f}s")

            for i, (recog_result, person_box) in enumerate(zip(recognition_results_raw, (box for _, box in recognition_tasks))):
                if isinstance(recog_result, Exception):
                    logger.error(f"Frame #{current_frame_num}: Recognition failed for person #{i+1}: {recog_result}", exc_info=recog_result)
                    continue
                is_felix, confidence = recog_result
                results.append({
                    "box": [int(c) for c in person_box],
                    "is_felix": bool(is_felix),
                    "confidence": float(confidence)
                })
            processing_duration = time.time() - processing_start_time
            logger.info(f"----- Finished Frame #{current_frame_num} Proc in {processing_duration:.4f}s ({len(results)} results) -----")
            return results

        except Exception as e:
            logger.error(f"Frame #{current_frame_num}: Unexpected error in process_frame: {e}", exc_info=True)
            return []


    async def handle_client(self, websocket):
        """Handle client connection: receive frame, process, visualize, send annotated frame."""
        client_id = websocket.remote_address
        logger.info(f"+++ Client connected: {client_id} +++")
        message_count = 0
        try:
            async for message in websocket: # Expecting JPEG bytes from client
                message_count += 1
                message_start_time = time.monotonic()
                logger.debug(f"Client {client_id}: Received message #{message_count} (size: {len(message)} bytes)")

                original_frame = None
                processed_results = []
                annotated_frame = None
                encoded_annotated_bytes = None

                try:
                    # --- 1. Decode received frame (non-blocking) ---
                    decode_start_time = time.monotonic()
                    try:
                         frame_bytes_np = np.frombuffer(message, dtype=np.uint8)
                         original_frame = await asyncio.to_thread(cv2.imdecode, frame_bytes_np, cv2.IMREAD_COLOR)
                    except Exception as decode_err:
                         logger.error(f"Client {client_id}, Msg #{message_count}: Error during decode: {decode_err}", exc_info=True)
                         original_frame = None
                    decode_duration = time.monotonic() - decode_start_time
                    logger.debug(f"Client {client_id}, Msg #{message_count}: Decode finished in {decode_duration:.4f}s")

                    if original_frame is None:
                        logger.error(f"Cliient {client_id}, Msg #{message_count}: Could not decode frame. Skipping.")
                        continue # Skip processing if decode failed

                    # --- 2. Process frame for detections (non-blocking ML) ---
                    self.global_frame_count += 1 # Increment global counter
                    processed_results = await self.process_frame(original_frame, self.global_frame_count)

                    # --- 3. Visualize results onto the frame (non-blocking drawing) ---
                    viz_start_time = time.monotonic()
                    try:
                         # Pass original frame and results to visualization func in thread
                         annotated_frame = await asyncio.to_thread(self.visualize_server_side, original_frame, processed_results)
                    except Exception as viz_err:
                         logger.error(f"Client {client_id}, Msg #{message_count}: Error during visualization: {viz_err}", exc_info=True)
                         annotated_frame = original_frame # Send original on viz error
                    viz_duration = time.monotonic() - viz_start_time
                    logger.debug(f"Client {client_id}, Msg #{message_count}: Visualize finished in {viz_duration:.4f}s")

                    # --- 4. Encode annotated frame to send back (non-blocking) ---
                    if annotated_frame is not None:
                        encode_start_time = time.monotonic()
                        try:
                             # Encode with reasonable quality for display
                             params = [cv2.IMWRITE_JPEG_QUALITY, 90]
                             encode_result = await asyncio.to_thread(cv2.imencode, ".jpg", annotated_frame, params)
                             success, buffer = encode_result
                             if success:
                                 encoded_annotated_bytes = buffer.tobytes()
                             else:
                                  logger.error(f"Client {client_id}, Msg #{message_count}: Failed to encode annotated frame.")
                        except Exception as enc_err:
                             logger.error(f"Client {client_id}, Msg #{message_count}: Error during annotated frame encode: {enc_err}", exc_info=True)
                        encode_duration = time.monotonic() - encode_start_time
                        logger.debug(f"Client {client_id}, Msg #{message_count}: Annotated encode finished in {encode_duration:.4f}s")
                    else: # Should not happen if original_frame was valid
                         logger.error(f"Client {client_id}, Msg #{message_count}: Annotated frame is None before encoding.")


                    # --- 5. Send back the annotated frame bytes ---
                    if encoded_annotated_bytes:
                        send_start_time = time.monotonic()
                        await websocket.send(encoded_annotated_bytes)
                        send_duration = time.monotonic() - send_start_time
                        message_duration = time.monotonic() - message_start_time
                        logger.info(f"Client {client_id}, Msg #{message_count}: Sent annotated frame ({len(encoded_annotated_bytes)} bytes). Total cycle: {message_duration:.4f}s")
                    else:
                         logger.warning(f"Client {client_id}, Msg #{message_count}: No annotated frame bytes to send.")
                         # Consider sending an empty message or error indicator?


                except websockets.exceptions.ConnectionClosed:
                     logger.warning(f"Client {client_id}: Connection closed during message processing.")
                     break # Exit async for loop
                except Exception as e:
                    # Catch errors from process_frame, visualize, encode, send
                    logger.error(f"Client {client_id}, Msg #{message_count}: Error processing message cycle: {e}", exc_info=True)
                    # Maybe attempt to send an error indicator? For now, just continue/log.


        except websockets.exceptions.ConnectionClosedOK:
            logger.info(f"Client {client_id}: Disconnected normally.")
        except websockets.exceptions.ConnectionClosedError as e:
             logger.warning(f"Client {client_id}: Connection closed with error: {e}")
        except Exception as e:
            logger.error(f"Error handling client {client_id}: {e}", exc_info=True)
        finally:
            logger.info(f"--- Client {client_id} handler finished ---")


async def main():
    # ... (Your existing main setup logic: model paths, server creation) ...
    logger.info("Starting main function...")

    # --- Configuration ---
    model_path = "/root/models/felix_classifier.pth" # CHANGE AS NEEDED
    yolo_path = None # Use default YOLOv8x
    server_port = 8080
    ping_interval_s = 15
    ping_timeout_s = 45 # Increased timeout
    # ---------------------

    if not os.path.exists(model_path):
        logger.critical(f"FATAL: Felix model file not found at {model_path}")
        return

    try:
        logger.info("Creating server instance...")
        server = FelixDetectionServer(
            felix_model=model_path,
            yolo_model=yolo_path
        )
        logger.info("Server instance created successfully")

        logger.info(f"Starting websocket server on port {server_port}...")
        async with websockets.serve(
            server.handle_client,
            "0.0.0.0",
            server_port,
            ping_interval=ping_interval_s,
            ping_timeout=ping_timeout_s,
            max_size=10*1024*1024 # Adjust if needed
        ):
            logger.info(f"=== Server running on ws://0.0.0.0:{server_port} ===")
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