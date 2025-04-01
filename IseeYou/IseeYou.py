# File: IseeYou.py
# --- REFACTOR: Added file description ---
"""
Felix Tracking Client: Connects to the GPUserver, sends video frames,
receives detection/recognition results, performs tracking, and displays
the annotated video feed. Includes logic to start/stop tracking on demand.
"""

import asyncio
import websockets
import cv2
import numpy as np
import json
import time
import supervision as sv
import traceback
import os
import sys # --- REFACTOR: Added sys ---
import logging # --- REFACTOR: Added logging ---
from typing import Optional, List, Dict, Any, Union # --- REFACTOR: Updated typing ---

# --- REFACTOR: Configure logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')

# --- REFACTOR: Import config ---
try:
    # Assuming config.py is in the parent directory relative to this file's location
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import config
except ImportError as e:
     logging.error(f"Error importing config module: {e}. Ensure config.py exists.", exc_info=True)
     # Use defaults if config import fails, or exit
     class config: # Dummy class with defaults
         FELIX_SERVER_URL = "ws://localhost:8080"
         FELIX_VIDEO_SOURCE = "0"
         # Add other needed defaults for tracker params if not hardcoded
     logging.warning("Using default configuration values as config.py could not be imported.")
     # Or sys.exit(1)


class FelixTrackingClient:
    """
    Handles video capture, communication with the detection server,
    object tracking (ByteTrack), and visualization. Allows starting and
    stopping the tracking process dynamically.

    Attributes:
        server_url (str): WebSocket URL of the FelixDetectionServer.
        video_source (Union[int, str]): Camera index or video file path.
        current_frame (Optional[np.ndarray]): The latest captured frame.
        frame_lock (asyncio.Lock): Lock for accessing current_frame.
        tracked_detections (Optional[sv.Detections]): Latest tracking results.
        byte_tracker (Optional[sv.ByteTrack]): ByteTrack instance.
        box_annotator (sv.BoxAnnotator): For drawing bounding boxes.
        label_annotator (sv.LabelAnnotator): For drawing labels.
        _is_running (bool): Global flag to control client operation (used for shutdown).
        _tracking_active (bool): Flag indicating if capture/receive/display tasks are running.
        _websocket (Optional[websockets.WebSocketClientProtocol]): Active WebSocket connection.
        _tasks (List[asyncio.Task]): List to keep track of running asyncio tasks.
        _connection_lock (asyncio.Lock): Lock to prevent concurrent connection/disconnection.
        frame_count (int): Counter for processed frames during tracking session.
    """

    def __init__(self, server_url: str = config.FELIX_SERVER_URL, video_source: Optional[Union[int, str]] = None):
        """
        Initializes the FelixTrackingClient.

        Args:
            server_url (str): WebSocket URL of the detection server.
            video_source (Optional[Union[int, str]]): Default video source (camera index or file path).
                                                     Can be overridden in start_tracking.
                                                     Defaults to config.FELIX_VIDEO_SOURCE.
        """
        logging.info("Initializing FelixTrackingClient...")
        self.server_url = server_url

        # --- REFACTOR: Set video source, allow override later ---
        if video_source is None:
            vid_src_str = config.FELIX_VIDEO_SOURCE
            try:
                 # Try converting to int first for camera index
                 self.video_source: Union[int, str] = int(vid_src_str)
            except ValueError:
                 # If not an int, assume it's a file path or string identifier
                 self.video_source: Union[int, str] = vid_src_str
        else:
             self.video_source: Union[int, str] = video_source
        logging.info(f"Default video source set to: {self.video_source}")


        # Shared state
        self.current_frame: Optional[np.ndarray] = None
        self.frame_lock = asyncio.Lock()
        self.tracked_detections: Optional[sv.Detections] = None
        self._is_running: bool = True # Overall client running flag (for graceful shutdown)
        self._tracking_active: bool = False # Flag for tracking tasks specifically
        self._websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._tasks: List[asyncio.Task] = []
        self._connection_lock = asyncio.Lock() # Prevent race conditions on connect/disconnect
        self.frame_count: int = 0 # Frame counter for tracking updates

        # Tracking and Visualization components
        self.byte_tracker: Optional[sv.ByteTrack] = None
        self.box_annotator: Optional[sv.BoxAnnotator] = None
        self.label_annotator: Optional[sv.LabelAnnotator] = None
        self._initialize_tracking_components() # Initialize tracker and annotators

        logging.info(f"FelixTrackingClient initialized for server: {self.server_url}")

    def _initialize_tracking_components(self):
        """Initializes ByteTrack and Supervision annotators."""
        logging.info("Initializing ByteTrack and Annotators...")
        try:
            # --- REFACTOR: Use parameters from config or defaults ---
            # Example using defaults, move to config if needed
            self.byte_tracker = sv.ByteTrack(
                track_activation_threshold=0.25,
                lost_track_buffer=30,
                minimum_matching_threshold=0.8,
                frame_rate=30 # Adjust if known, otherwise estimate
            )
            # --- REFACTOR: Initialize annotators here ---
            self.box_annotator = sv.BoxAnnotator(thickness=2)
            self.label_annotator = sv.LabelAnnotator(
                text_position=sv.Position.TOP_CENTER,
                text_scale=0.5,
                text_thickness=1,
                text_padding=2,
            )
            logging.info("ByteTrack and Annotators initialized successfully.")
        except TypeError as e:
            # Catch specific errors related to Supervision versions/parameters
            logging.error(f"CRITICAL ERROR initializing ByteTrack (TypeError): {e}", exc_info=True)
            logging.error("Please check supervision library version and ByteTrack parameters.")
            self.byte_tracker = None
        except Exception as e:
            # --- REFACTOR: Log critical error, ByteTrack is essential ---
            logging.error(f"CRITICAL ERROR initializing ByteTrack or Annotators: {e}", exc_info=True)
            logging.error("Tracking and visualization will not function.")
            self.byte_tracker = None # Mark as failed
            # Annotators might still work partially, but check usage


    async def _connect(self) -> bool:
        """Establishes WebSocket connection."""
        # --- REFACTOR: Added lock, better logging, more error types ---
        async with self._connection_lock: # Ensure only one connection attempt at a time
            if self._websocket and self._websocket.open:
                logging.debug("WebSocket connection already open.") # Use debug for less noise
                return True
            if not self._is_running:
                 logging.warning("Client is shutting down, connection aborted.")
                 return False

            logging.info(f"Attempting to connect to server: {self.server_url}")
            try:
                # Connect with timeouts
                self._websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.server_url,
                        ping_interval=20,
                        ping_timeout=20,
                        # Increase max size if needed, matches server
                        max_size=10 * 1024 * 1024
                    ),
                    timeout=10.0 # Connection timeout
                )
                logging.info(f"WebSocket connection established successfully to {self.server_url}.")
                return True
            except asyncio.TimeoutError:
                logging.error(f"Connection timed out attempting to reach {self.server_url}.")
            except websockets.exceptions.InvalidURI:
                logging.error(f"Invalid WebSocket URI: {self.server_url}")
            except websockets.exceptions.WebSocketException as e: # General WS errors
                logging.error(f"WebSocket connection failed ({self.server_url}): {e}")
            except ConnectionRefusedError:
                 logging.error(f"Connection refused by server ({self.server_url}). Is the detection server running?")
            except OSError as e: # Includes address in use, network errors
                logging.error(f"Network error during connection ({self.server_url}): {e}")
            except Exception as e:
                logging.error(f"Unexpected error during connection ({self.server_url}): {e}", exc_info=True)

            self._websocket = None # Ensure websocket is None on failure
            return False

    async def _disconnect(self):
        """Closes WebSocket connection."""
        # --- REFACTOR: Added lock ---
        async with self._connection_lock:
            ws = self._websocket # Local variable for safety
            if ws and not ws.closed:
                logging.info("Closing WebSocket connection.")
                try:
                    await ws.close()
                except websockets.exceptions.WebSocketException as e:
                    logging.warning(f"Error closing WebSocket: {e}")
                except Exception as e:
                     logging.warning(f"Unexpected error closing WebSocket: {e}")
                finally:
                     # Ensure instance variable is cleared even if close fails
                     if self._websocket == ws: # Check if it hasn't changed in the meantime
                          self._websocket = None
            # else: # Can be noisy
            #      logging.debug("WebSocket connection already closed or not established.")
            # Explicitly clear instance variable outside the if block for robustness
            self._websocket = None


    async def _cancel_tasks(self):
        """Cancels all running asyncio tasks."""
        # --- REFACTOR: Improved safety and logging ---
        tasks_to_cancel = list(self._tasks) # Create copy to iterate over
        if not tasks_to_cancel:
            return
        logging.info(f"Cancelling {len(tasks_to_cancel)} running tasks...")
        for task in tasks_to_cancel:
            if not task.done():
                task.cancel()
        try:
            # Wait for tasks to actually cancel
            # Use timeout to prevent hanging indefinitely
            await asyncio.wait_for(asyncio.gather(*tasks_to_cancel, return_exceptions=True), timeout=5.0)
            logging.info("Tasks cancelled.")
        except asyncio.TimeoutError:
             logging.warning("Timeout waiting for tasks to cancel. Some tasks might linger.")
        except asyncio.CancelledError:
             logging.info("Task cancellation itself was cancelled (during shutdown).")
        except Exception as e:
            logging.error(f"Error during task cancellation gathering: {e}", exc_info=True)
        finally:
            self._tasks = [] # Clear the original task list


    # --- REFACTOR: Logic to start tracking tasks ---
    async def start_tracking(self, video_source: Optional[Union[int, str]] = None) -> bool:
        """
        Connects to the server and starts video capture, receiving, and display tasks.

        Args:
            video_source (Optional[Union[int, str]]): Override the default video source.

        Returns:
            bool: True if tracking started successfully, False otherwise.
        """
        if self._tracking_active:
            logging.warning("Tracking is already active.")
            return True
        if not self._is_running:
            logging.warning("Cannot start tracking, client is shutting down.")
            return False
        if self.byte_tracker is None:
             logging.error("Cannot start tracking: ByteTrack failed to initialize.")
             return False

        # Use provided video source or default
        current_video_source = video_source if video_source is not None else self.video_source
        logging.info(f"--- Starting Tracking using source: {current_video_source} ---")

        # Attempt to connect
        if not await self._connect():
            logging.error("Failed to connect to server. Cannot start tracking.")
            return False

        # Reset frame count for new session
        self.frame_count = 0
        self.tracked_detections = None # Clear previous tracking results

        # Create and store tasks
        logging.info("Creating tracking tasks (capture, receive, display)...")
        try:
            # Ensure websocket connection exists before creating tasks using it
            if not self._websocket or not self._websocket.open: # Check if open
                 logging.error("WebSocket not connected or not open, aborting task creation.")
                 await self._disconnect() # Clean up potential partial connection
                 return False

            capture_task = asyncio.create_task(
                self._capture_and_send_frames(self._websocket, current_video_source),
                name="capture_task"
            )
            receive_task = asyncio.create_task(
                self._receive_results(self._websocket),
                name="receive_task"
            )
            display_task = asyncio.create_task(
                self._display_loop(),
                name="display_task"
            )
            self._tasks = [capture_task, receive_task, display_task]
            self._tracking_active = True
            logging.info("Tracking tasks started.")

            # Start a background task to monitor the main tracking tasks
            monitor = asyncio.create_task(self._monitor_tracking_tasks(), name="monitor_task")
            self._tasks.append(monitor)

            return True

        except Exception as e:
             logging.error(f"Error creating tracking tasks: {e}", exc_info=True)
             await self._stop_tracking_internal() # Attempt cleanup if task creation failed
             return False

    async def _monitor_tracking_tasks(self):
         """Monitors the main tracking tasks and triggers stop if one fails or completes."""
         # --- REFACTOR: More robust monitoring ---
         if not self._tasks: return
         # Get only the core tasks (exclude monitor itself)
         core_tasks = [t for t in self._tasks if t.get_name() != "monitor_task"]
         if not core_tasks:
              logging.warning("Monitor task found no core tasks to monitor.")
              return

         try:
              # Wait for any of the core tasks to complete
              done, pending = await asyncio.wait(core_tasks, return_when=asyncio.FIRST_COMPLETED)

              # If we are here, at least one task finished. Log details.
              logging.warning("--- A core tracking task finished ---")
              for task in done:
                  task_name = task.get_name() or "Unknown Task"
                  try:
                       # Check if the task raised an exception
                       exc = task.exception()
                       if exc:
                            logging.error(f"Task '{task_name}' failed: {exc}", exc_info=exc)
                       else:
                            logging.info(f"Task '{task_name}' completed normally (but unexpectedly).")
                  except asyncio.CancelledError:
                       logging.info(f"Task '{task_name}' was cancelled.")
                  except Exception as e:
                       logging.error(f"Error retrieving exception from task '{task_name}': {e}")


              # If any task finishes (error or not), stop all other tracking tasks gracefully
              if self._tracking_active: # Avoid triggering stop multiple times
                   logging.warning("Triggering tracking stop due to task completion/failure.")
                   # Don't await here to avoid potential deadlocks if called from within a task
                   asyncio.create_task(self._stop_tracking_internal())

         except asyncio.CancelledError:
              logging.info("Tracking monitor task cancelled.")
         except Exception as e:
              logging.error(f"Error in tracking monitor task: {e}", exc_info=True)
              # Optionally trigger stop here too if monitor itself fails
              if self._tracking_active:
                   logging.error("Triggering tracking stop due to monitor task failure.")
                   asyncio.create_task(self._stop_tracking_internal())


    # --- REFACTOR: Logic to stop tracking tasks ---
    async def stop_tracking(self):
        """Stops the video capture, receiving, and display tasks and disconnects."""
        # --- REFACTOR: Public method calls internal one ---
        logging.info("--- Received request to stop tracking ---")
        await self._stop_tracking_internal()

    async def _stop_tracking_internal(self):
         """Internal implementation to stop tracking tasks and disconnect."""
         # --- REFACTOR: Add lock, improve state handling ---
         async with self._connection_lock: # Prevent racing with start/connect
             if not self._tracking_active:
                 logging.info("Tracking is not currently active.")
                 # Ensure connection is closed even if tasks weren't active
                 await self._disconnect() # Call disconnect directly
                 return

             self._tracking_active = False # Set flag immediately

             # Cancel tasks
             await self._cancel_tasks()

             # Disconnect websocket (now handled within _disconnect)
             await self._disconnect()

             # Reset shared state safely
             async with self.frame_lock:
                 self.current_frame = None
             self.tracked_detections = None

             # Close OpenCV window if it's open
             # This needs to be done carefully, maybe signal display loop?
             # For now, rely on display loop's finally block.
             # cv2.destroyAllWindows() # Might cause issues if called from non-main thread

             logging.info("Tracking tasks stopped and disconnected.")


    # --- REFACTOR: Renamed to internal methods, improved logging/error handling ---
    async def _capture_and_send_frames(self, websocket: websockets.WebSocketClientProtocol, video_source: Union[int, str]):
        """(Internal) Captures frames and sends them to the server."""
        logging.info(f"=== Starting video capture from: {video_source} ===")
        cap = None
        frame_counter = 0
        last_print_time = time.time()
        consecutive_read_errors = 0
        max_read_errors = 10 # Stop after this many consecutive read failures

        try:
            # --- REFACTOR: Test capture source before loop ---
            try:
                cap = cv2.VideoCapture(video_source)
                if not cap.isOpened():
                    raise IOError(f"Cannot open video source: {video_source}")
                # Set buffer size to minimum to get fresh frames (optional)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                ret, test_frame = cap.read()
                if not ret or test_frame is None:
                    raise IOError(f"Cannot read initial frame from video source: {video_source}")
                h, w = test_frame.shape[:2]
                logging.info(f"Video source {video_source} opened successfully (Resolution: {w}x{h})")
                del test_frame # Release memory
            except (cv2.error, IOError, Exception) as e:
                logging.error(f"Failed to initialize video source '{video_source}': {e}", exc_info=True)
                # Signal failure by stopping tracking
                asyncio.create_task(self._stop_tracking_internal()) # Schedule stop
                return # Exit task

            while self._tracking_active and self._is_running: # Check both flags
                ret, frame = cap.read()

                if not ret or frame is None:
                    consecutive_read_errors += 1
                    logging.warning(f"Cannot read frame (attempt {consecutive_read_errors}/{max_read_errors}). End of stream or hardware issue?")
                    if consecutive_read_errors >= max_read_errors:
                         logging.error("Maximum consecutive frame read errors reached. Stopping capture.")
                         break # Exit loop
                    await asyncio.sleep(0.5) # Wait before retrying
                    continue
                else:
                     consecutive_read_errors = 0 # Reset error count on success

                frame_counter += 1

                # Update shared frame for display task
                async with self.frame_lock:
                    # Make a copy to prevent race conditions if display modifies it (though it shouldn't)
                    self.current_frame = frame.copy()

                # Encode frame (use JPEG for reasonable compression)
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85] # Quality 0-100
                result, encoded_frame = cv2.imencode(".jpg", frame, encode_param)
                if not result:
                    logging.warning("Couldn't encode frame, skipping send.")
                    continue

                frame_bytes = encoded_frame.tobytes()

                # Send frame via WebSocket
                try:
                    # Send with timeout
                    await asyncio.wait_for(websocket.send(frame_bytes), timeout=5.0)
                except asyncio.TimeoutError:
                    logging.warning("Timeout sending frame to server. Connection might be lagging.")
                    # Consider breaking or implementing more robust error handling
                except websockets.exceptions.ConnectionClosed:
                    logging.warning("WebSocket connection closed during send. Exiting capture loop.")
                    break # Exit loop
                except Exception as e:
                     # Log other send errors but try to continue
                     logging.error(f"Error sending frame: {e}", exc_info=True)
                     await asyncio.sleep(0.1) # Short pause after error

                # Print stats periodically
                current_time = time.time()
                if current_time - last_print_time >= 10.0: # Print every 10 seconds
                    logging.info(f"[Capture] Sent {frame_counter} frames.")
                    last_print_time = current_time

                # Control frame rate - yield control briefly
                await asyncio.sleep(0.01) # Allows other tasks to run

        except asyncio.CancelledError:
            logging.info("[Capture] Task cancelled.")
        except Exception as e:
            logging.error(f"Unhandled error in _capture_and_send_frames: {e}", exc_info=True)
        finally:
            if cap and cap.isOpened():
                cap.release()
                logging.info("Video source released.")
            logging.info("[Capture] Capture task finished.")
            # Ensure tracking stops if capture fails critically or finishes
            if self._tracking_active:
                 logging.warning("Capture task ended, ensuring tracking stops.")
                 asyncio.create_task(self._stop_tracking_internal())


    async def _receive_results(self, websocket: websockets.WebSocketClientProtocol):
        """(Internal) Receives detection results from the server."""
        logging.info("=== Starting to receive detection results ===")
        detection_counter = 0
        try:
            async for message in websocket:
                # --- Completion of the method ---
                if not self._tracking_active or not self._is_running: break # Check flags

                detection_counter += 1
                try:
                    # Ensure message is string before JSON parsing
                    if isinstance(message, bytes):
                         message_str = message.decode('utf-8')
                    elif isinstance(message, str):
                         message_str = message
                    else:
                         logging.warning(f"Received unexpected message type: {type(message)}. Skipping.")
                         continue

                    # Parse the JSON message
                    detection_data = json.loads(message_str)

                    # Validate data format (basic check)
                    if not isinstance(detection_data, list):
                         logging.warning(f"Received non-list detection data: {detection_data}")
                         continue

                    # Log reception periodically
                    if detection_counter % 30 == 0: # Log every 30 frames
                        felix_count = sum(1 for det in detection_data if det.get("is_felix", False))
                        logging.info(f"[Receiver] Frame #{detection_counter}: Received {len(detection_data)} detections ({felix_count} Felix)")

                    # Update tracking state with new detections
                    self._update_tracking(detection_data) # Call internal tracking update

                except json.JSONDecodeError:
                    logging.error(f"[Receiver] Error: Received invalid JSON: {message_str[:200]}...") # Log start of message
                except websockets.exceptions.ConnectionClosed:
                     logging.warning("[Receiver] WebSocket connection closed while receiving.")
                     break # Exit loop
                except asyncio.CancelledError:
                     raise # Propagate cancellation
                except Exception as e:
                     logging.error(f"[Receiver] Error processing message: {e}", exc_info=True)

        except asyncio.CancelledError:
             logging.info("[Receiver] Task cancelled.")
        except websockets.exceptions.ConnectionClosedOK:
            logging.info("[Receiver] WebSocket connection closed normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            logging.error(f"[Receiver] WebSocket connection closed with error: {e}")
        except Exception as e:
            logging.error(f"[Receiver] Unhandled error in receive loop: {e}", exc_info=True)
        finally:
            logging.info("[Receiver] Receive task finished.")
            # Ensure tracking stops if receiver fails critically or finishes
            if self._tracking_active:
                 logging.warning("Receive task ended, ensuring tracking stops.")
                 asyncio.create_task(self._stop_tracking_internal())


    def _update_tracking(self, detections: List[Dict[str, Any]]):
        """(Internal) Updates ByteTrack with new detections from the server."""
        # --- REFACTOR: Renamed, improved safety checks ---
        if not self.byte_tracker:
            # Log less frequently if tracker is missing
            if self.frame_count % 120 == 0: # Log every ~4 seconds if tracker missing
                logging.error("[Tracking] Update skipped: ByteTrack not initialized.")
            return # Safety check

        self.frame_count += 1 # Increment frame counter used by tracker

        # Convert received detections to supervision.Detections object
        boxes_xyxy = []
        confidences = []
        class_ids = [] # 0 = Felix, 1 = Not Felix (consistent definition needed)

        for det in detections:
            try:
                # Validate detection structure
                box = det.get("box")
                conf = det.get("confidence")
                is_felix = det.get("is_felix")

                if box is None or conf is None or is_felix is None or len(box) != 4:
                     logging.warning(f"[Tracking] Skipping invalid detection data format: {det}")
                     continue

                x, y, w, h = map(int, box) # Ensure integer coords

                # Convert [x, y, w, h] to [x1, y1, x2, y2] for Supervision
                x1, y1, x2, y2 = x, y, x + w, y + h
                boxes_xyxy.append([x1, y1, x2, y2])
                confidences.append(float(conf))
                class_ids.append(0 if is_felix else 1) # Assign class ID

            except (KeyError, TypeError, ValueError) as e:
                logging.warning(f"[Tracking] Skipping invalid detection data content: {det}, Error: {e}")
                continue # Skip this detection

        # Create supervision.Detections object
        if not boxes_xyxy: # If no valid detections received
            sv_detections = sv.Detections.empty()
        else:
            try:
                sv_detections = sv.Detections(
                    xyxy=np.array(boxes_xyxy, dtype=np.float32),
                    confidence=np.array(confidences, dtype=np.float32),
                    class_id=np.array(class_ids, dtype=int)
                )
            except Exception as e:
                 logging.error(f"[Tracking] Error creating sv.Detections object: {e}", exc_info=True)
                 sv_detections = sv.Detections.empty() # Use empty on error

        # Update ByteTrack
        try:
            # Store the tracked results
            self.tracked_detections = self.byte_tracker.update_with_detections(sv_detections)
        except Exception as e:
             logging.error(f"[Tracking] Error during byte_tracker.update: {e}", exc_info=True)
             self.tracked_detections = None # Reset tracked_detections on error


    def _visualize_frame(self, frame: np.ndarray) -> np.ndarray:
        """(Internal) Draws tracking results onto the frame."""
        # --- REFACTOR: Renamed, improved checks and drawing logic ---
        if frame is None:
            logging.warning("[Visualize] Received None frame.")
            # Return a placeholder or raise error? Return placeholder for now.
            return np.zeros((480, 640, 3), dtype=np.uint8) # Example placeholder

        # Ensure annotators are available
        if not self.box_annotator or not self.label_annotator:
             if self.frame_count % 120 == 0: # Log infrequently
                  logging.warning("[Visualize] Annotators not initialized, returning raw frame.")
             return frame.copy() # Return original frame if no annotators

        # Get the latest tracked detections safely
        # Make a local copy to prevent modification during iteration if updates happen concurrently (though unlikely with asyncio locks)
        current_tracks = self.tracked_detections

        # If no tracks or tracking failed, return the original frame
        if current_tracks is None or len(current_tracks) == 0:
            # logging.debug("[Visualize] No tracks to visualize") # Use debug level
            return frame.copy()

        try:
            # Make a copy of the frame to draw on
            annotated_frame = frame.copy()

            # Define colors based on class ID (0=Felix, 1=Not Felix)
            # Use Supervision colors
            color_map = {0: sv.Color.GREEN, 1: sv.Color.RED}
            colors_for_detections = [color_map.get(cid, sv.Color.WHITE) for cid in current_tracks.class_id]

            # Create labels for the label annotator
            labels = []
            for i, track_id in enumerate(current_tracks.tracker_id):
                class_id = current_tracks.class_id[i]
                conf = current_tracks.confidence[i] if current_tracks.confidence is not None else 0.0
                person_type = "Felix" if class_id == 0 else "Person" # Changed "Not Felix" to "Person"
                labels.append(f"ID:{track_id} {person_type} {conf:.2f}")

            # Apply box annotations with specific colors
            # Note: Supervision's BoxAnnotator might not directly support per-detection colors easily.
            # We can annotate all first, then redraw rectangles with correct colors if needed.
            # Or, iterate and call annotate for each detection (less efficient).
            # Let's try the standard annotate and rely on LabelAnnotator for color hints.

            annotated_frame = self.box_annotator.annotate(
                 scene=annotated_frame,
                 detections=current_tracks,
                 # colors=colors_for_detections, # Check if `colors` arg exists and works
                 # skip_label=True # Skip default box labels if using LabelAnnotator
            )

            # Apply label annotations (which can use color)
            annotated_frame = self.label_annotator.annotate(
                scene=annotated_frame,
                detections=current_tracks,
                labels=labels,
                # colors=colors_for_detections # Pass colors to LabelAnnotator as well
            )

            # --- Manual Re-coloring (if BoxAnnotator didn't support per-detection colors) ---
            # Uncomment this section if boxes don't have the right colors
            # for i, xyxy in enumerate(current_tracks.xyxy):
            #      x1, y1, x2, y2 = map(int, xyxy)
            #      color_bgr = colors_for_detections[i].as_bgr() # Get BGR tuple for cv2
            #      cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color_bgr, self.box_annotator.thickness)
            # --- End Manual Re-coloring ---

            return annotated_frame

        except Exception as e:
            logging.error(f"[Visualize] Error during visualization: {e}", exc_info=True)
            # Return the original frame on error to avoid crashing display
            return frame.copy()


    async def _display_loop(self):
        """(Internal) Displays processed frames in an OpenCV window."""
        logging.info("=== Starting display loop ===")
        window_name = 'Felix Tracking Client'
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO) # Make resizable, keep aspect ratio
            cv2.resizeWindow(window_name, 1280, 720) # Example starting size
        except cv2.error as e:
             logging.error(f"Error creating OpenCV window '{window_name}': {e}. Display disabled.")
             # Signal stop? Or just disable display? Disable for now.
             asyncio.create_task(self._stop_tracking_internal()) # Stop if display fails
             return

        try:
            while self._tracking_active and self._is_running: # Check flags
                frame_to_display = None
                # Get the current frame safely
                async with self.frame_lock:
                    if self.current_frame is not None:
                        # Make a copy for visualization to avoid modifying shared frame
                        frame_to_display = self.current_frame.copy()

                if frame_to_display is None:
                    # logging.debug("[Display] No frame available yet.") # Debug level
                    await asyncio.sleep(0.02) # Wait briefly if no frame
                    continue

                # Annotate the frame
                processed_frame = self._visualize_frame(frame_to_display)

                # Display the frame
                try:
                    cv2.imshow(window_name, processed_frame)
                except cv2.error as e:
                     logging.error(f"Error displaying frame in window '{window_name}': {e}")
                     # If display fails consistently, maybe break? For now, continue.
                     await asyncio.sleep(0.1)


                # Check for key press ('q' to quit) - crucial for stopping
                key = cv2.waitKey(1) & 0xFF # Check every 1ms
                if key == ord('q'):
                    logging.info("[Display] 'q' pressed. Signaling stop...")
                    # Don't call stop_tracking directly from here if it awaits, schedule it
                    asyncio.create_task(self.stop_tracking()) # Use public method
                    break # Exit display loop immediately

                # Yield control briefly to allow other tasks to run
                await asyncio.sleep(0.01)

        except asyncio.CancelledError:
             logging.info("[Display] Task cancelled.")
        except Exception as e:
            logging.error(f"Error in display_loop: {e}", exc_info=True)
        finally:
            try:
                cv2.destroyWindow(window_name)
                # Add extra waitKey calls to ensure window closes on all OS
                for _ in range(5): cv2.waitKey(1)
            except cv2.error as e:
                 logging.warning(f"Error destroying OpenCV window '{window_name}': {e}")
            except Exception as e:
                 logging.warning(f"Unexpected error during display cleanup: {e}")
            logging.info("[Display] Display task finished.")
            # Ensure tracking stops if display fails critically or finishes
            if self._tracking_active:
                 logging.warning("Display task ended, ensuring tracking stops.")
                 asyncio.create_task(self._stop_tracking_internal())


    # --- REFACTOR: Renamed from stop() to shutdown() for clarity ---
    def shutdown(self):
        """Signals the client to stop all operations and exit gracefully."""
        logging.info("--- Initiating FelixTrackingClient shutdown ---")
        self._is_running = False # Signal all loops to stop
        # Schedule stop_tracking to run in the event loop
        try:
             loop = asyncio.get_running_loop()
             loop.create_task(self.stop_tracking())
        except RuntimeError:
             logging.warning("No running event loop found during shutdown signal.")
             # Try running stop_tracking directly if no loop (might block if called from sync code)
             # asyncio.run(self.stop_tracking()) # Be cautious with this


# --- REFACTOR: Standalone execution part adapted ---
async def _standalone_main():
    """Runs the client for standalone testing."""
    print("Starting Felix Tracking Client (Standalone Test)...")
    # Use configuration for defaults
    server_url = config.FELIX_SERVER_URL
    video_source_config = config.FELIX_VIDEO_SOURCE
    try:
        video_source = int(video_source_config)
    except ValueError:
        video_source = video_source_config # Use as file path if not integer

    print(f"Server URL: {server_url}")
    print(f"Video Source: {video_source}")

    client = FelixTrackingClient(server_url=server_url)
    main_task = None

    try:
        # Start tracking
        started = await client.start_tracking(video_source=video_source)

        if started:
            print("Tracking started successfully. Press 'q' in the display window to stop.")
            # Keep running until shutdown signal (_is_running becomes False)
            while client._is_running:
                 await asyncio.sleep(1)
            print("Client shutdown signal received in standalone main.")
        else:
             print("Failed to start tracking.")

    except asyncio.CancelledError:
        print("Standalone main task cancelled.")
    except KeyboardInterrupt:
         print("\nCtrl+C detected in standalone main. Initiating shutdown...")
    except Exception as e:
         print(f"Unhandled exception in standalone main: {e}")
         traceback.print_exc()
    finally:
        print("Standalone main: Cleaning up...")
        # Ensure shutdown is called to stop tasks and close connections
        if 'client' in locals():
            client.shutdown()
            # Allow some time for cleanup tasks to run
            await asyncio.sleep(2)
        print("Standalone main: Cleanup finished.")


if __name__ == "__main__":
    try:
        # --- REFACTOR: Use asyncio.run for cleaner execution ---
        asyncio.run(_standalone_main())
    except KeyboardInterrupt:
        # This might not be reached if caught inside _standalone_main
        print("\nProgram interrupted by user (Ctrl+C) at top level.")
    except Exception as e:
        print(f"Unhandled exception during standalone execution: {e}")
        traceback.print_exc()
    finally:
        print("Standalone execution finished.")

# --- END OF REFINED FILE IseeYou.py ---