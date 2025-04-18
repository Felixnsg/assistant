import asyncio
import websockets
import cv2
import numpy as np
import json
import time
import supervision as sv
import traceback
import logging

class FelixTrackingClient:
    """Handles video capture, tracking and visualization"""
    
    def __init__(self, server_url="ws://localhost:8080"):
        self.logger = logging.getLogger("IseeYou.FelixTrackingClient")
        self.logger.info("\n=== Initializing FelixTrackingClient ===")
        self.server_url = server_url
        
        self.frame_count = 0
        self.tracking_cache = {}
        # For sharing the current frame between tasks
        self.current_frame = None
        self.frame_lock = asyncio.Lock()
        
        # For fallback visualization when tracking doesn't work
        self.raw_detections = []
        
        # Skip ByteTrack - use our own simple tracking
        self.logger.info("Using simple direct tracking (no ByteTrack dependency)")
        
        # This will draw boxes, labels and tracking IDs on the frame
        self.box_annotator = sv.BoxAnnotator(
            thickness=3
        )
        
        # Reduce thresholds to make tracking more sensitive
        self.byte_tracker = sv.ByteTrack(
            track_activation_threshold=0.3,  # Lowered from 0.5
            lost_track_buffer=20,
            minimum_matching_threshold=0.5,  # Lowered from 0.7
            frame_rate=30
        )
        
        self.logger.info("FelixTrackingClient initialized successfully")
    
    async def capture_and_send_frames(self, websocket, video_source):
        """Capture frames, encode NON-BLOCKINGLY, and send to the server."""
        self.logger.info("\n=== Starting video capture ===")
        cap = cv2.VideoCapture(video_source)

        if not cap.isOpened():
            self.logger.error("Error: Cannot access camera")
            return

        try:
            frame_counter = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Error: Cannot read from camera")
                    await asyncio.sleep(0.1)
                    continue

                frame_counter += 1

                # Update the current frame for display task
                async with self.frame_lock:
                    self.current_frame = frame.copy()

                # --- Encode frame in thread ---
                try:
                    encoded_result_tuple = await asyncio.to_thread(cv2.imencode, ".jpg", frame)
                    success_flag, encoded_frame_buffer = encoded_result_tuple
                except Exception as encode_error:
                    self.logger.error(f"Error during threaded cv2.imencode for frame {frame_counter}: {encode_error}", exc_info=True)
                    await asyncio.sleep(0.01) # Small delay before next attempt
                    continue # Skip this frame

                if not success_flag or encoded_frame_buffer is None:
                    self.logger.error(f"Error: cv2.imencode failed for frame {frame_counter}")
                    await asyncio.sleep(0.01) # Small delay
                    continue # Skip this frame

                frame_bytes = encoded_frame_buffer.tobytes()

                # --- Send the encoded frame ---
                try:
                    await websocket.send(frame_bytes)
                    if frame_counter % 100 == 0: # Log progress periodically
                        self.logger.debug(f"[CAPTURE] Sent frame {frame_counter} ({len(frame_bytes)} bytes)")

                except websockets.exceptions.ConnectionClosed:
                    self.logger.warning("Connection closed during send. Exiting capture loop.")
                    break # *** EXIT LOOP on closed connection ***
                except Exception as send_error:
                    self.logger.error(f"Error sending frame {frame_counter}: {send_error}", exc_info=True)
                    # Optional: Add a delay or break depending on error severity
                    await asyncio.sleep(0.5)
                    # Consider if you should 'continue' or 'break' here

                # --- Control loop rate ---
                # This sleep controls the capture rate primarily
                await asyncio.sleep(0.1) # Target ~30 FPS capture/display rate

        except asyncio.CancelledError:
            self.logger.info("Capture and send task cancelled.")
        except Exception as e:
            self.logger.error(f"Unhandled error in capture_and_send_frames loop: {e}", exc_info=True)
        finally:
            if cap.isOpened(): # Check if cap is still open before releasing
                cap.release()
            self.logger.info("Camera released.")
                    
        
    
    # Inside FelixTrackingClient class:

    async def receive_results(self, websocket):
        """Receive detection results and update tracking NON-BLOCKINGLY."""
        self.logger.info("\n=== Starting to receive detection results ===")
        detection_counter = 0
        try:
            async for message in websocket:
                detection_counter += 1
                update_task = None # Keep track of the update task

                try:
                    # Offload potential blocking json.loads (optional but safe)
                    detection_data = await asyncio.to_thread(json.loads, message)

                    # Store raw detections for fallback visualization
                    # This assignment is fast, no lock needed unless accessed elsewhere simultaneously
                    self.raw_detections = detection_data

                    # Log what we received
                    felix_count = sum(1 for det in detection_data if det.get("is_felix", False))
                    self.logger.debug(f"[RECEIVER] Frame #{detection_counter}: Received {len(detection_data)} detections ({felix_count} Felix)")

                    # --- Update tracking in a separate thread ---
                    # Create task but don't await here if you want receiving to continue ASAP
                    # However, awaiting ensures tracking is done before processing next message fully.
                    # Let's await to ensure self.tracked_detections is updated.
                    tracking_start_time = time.time()
                    self.logger.debug(f"[RECEIVER] Frame #{detection_counter}: Submitting tracking update...")
                    # Run the synchronous update_tracking function in a thread
                    await asyncio.to_thread(self.update_tracking, detection_data)
                    tracking_duration = time.time() - tracking_start_time
                    self.logger.debug(f"[RECEIVER] Frame #{detection_counter}: Tracking update finished in {tracking_duration:.4f}s")
                    # self.tracked_detections is now updated

                except json.JSONDecodeError as json_err:
                    self.logger.error(f"[RECEIVER] Frame #{detection_counter}: Failed to decode JSON: {json_err}")
                    # Continue to next message
                except Exception as e:
                    # Catch errors during the to_thread call or subsequent logging
                    self.logger.error(f"[RECEIVER] Frame #{detection_counter}: Error processing message or tracking: {e}", exc_info=True)
                    # Continue to next message

        # Handle connection closed exceptions outside the inner try/except
        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info("[RECEIVER] Connection closed normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            self.logger.warning(f"[RECEIVER] Connection closed with error: {e}")
        except asyncio.CancelledError:
            self.logger.info("[RECEIVER] Receive task cancelled.")
        except Exception as e:
            self.logger.error(f"[RECEIVER] Unhandled error in receive loop: {e}", exc_info=True)
            traceback.print_exc() # Keep traceback for unexpected errors
        
    def update_tracking(self, detections):
        """Tracking with Supervision's ByteTrack wrapper"""
        self.frame_count += 1

        if not detections:
            if self.frame_count % 30 == 0:
                self.logger.debug(f"[TRACKING] Frame #{self.frame_count}: No detections received")
            return

        # Debug: Print confidence values occasionally
        if self.frame_count % 30 == 0:
            confidences = [det["confidence"] for det in detections]
            self.logger.debug(f"[TRACKING] Debug: Detection confidences: {confidences}")

        # Step 1: Convert to supervision.Detections object
        boxes = []
        confidences = []
        class_ids = []  # 0 = Felix, 1 = Not Felix

        for det in detections:
            x, y, w, h = det["box"]
            conf = det["confidence"]
            is_felix = det["is_felix"]

            x1, y1, x2, y2 = x, y, x + w, y + h
            boxes.append([x1, y1, x2, y2])
            confidences.append(conf)
            class_ids.append(0 if is_felix else 1)

        sv_detections = sv.Detections(
            xyxy=np.array(boxes),
            confidence=np.array(confidences),
            class_id=np.array(class_ids)
        )

        # Step 2: Track using ByteTrack
        tracked_detections = self.byte_tracker.update_with_detections(sv_detections)

        # Step 3: Store tracked detections for visualization
        self.tracked_detections = tracked_detections

        if self.frame_count % 30 == 0:
            felix_count = sum(1 for cid in tracked_detections.class_id if cid == 0)
            self.logger.debug(f"[TRACKING] Frame #{self.frame_count}: Tracking {len(tracked_detections)} objects ({felix_count} Felix)")
            
            # Debug message if tracking creates no tracks
            if len(tracked_detections) == 0 and len(detections) > 0:
                self.logger.warning(f"[TRACKING] Warning: ByteTrack created 0 tracks from {len(detections)} detections.")
    
    def visualize_frame(self, frame):
        """Draw detection results on the frame, with fallback to raw detections if tracking fails"""
        if frame is None:
            return None
            
        frame_copy = frame.copy()
        
        # OPTION 1: Use tracked detections if available
        if hasattr(self, 'tracked_detections') and len(self.tracked_detections) > 0:
            try:
                # Log visualization details occasionally
                if self.frame_count % 30 == 0:
                    felix_count = sum(1 for cid in self.tracked_detections.class_id if cid == 0)
                    self.logger.debug(f"[VISUALIZE] Frame #{self.frame_count}: Fallback to {len(self.raw_detections)} raw detections ({felix_count} Felix)")

                # Create custom labels for each detection
                labels = [
                    f"{'Felix' if class_id == 0 else 'Not Felix'} #{tracker_id}: {conf:.2f}"
                    for tracker_id, conf, class_id in zip(
                        self.tracked_detections.tracker_id,
                        self.tracked_detections.confidence,
                        self.tracked_detections.class_id
                    )
                ]
                
                # Create box annotator with custom colors
                box_annotator = sv.BoxAnnotator(
                    thickness=2,
                    color_lookup=lambda class_id: (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for Felix, Red for others
                )
                
                # Draw boxes and labels
                frame_copy = box_annotator.annotate(
                    scene=frame_copy,
                    detections=self.tracked_detections,
                    labels=labels
                )
                
                return frame_copy
                
            except Exception as e:
                self.logger.error(f"[VISUALIZE] Error with tracked visualization: {e}")
                # Will fall through to fallback methods
        
        # OPTION 2: Fallback to raw detections if tracking doesn't work
        elif hasattr(self, 'raw_detections') and self.raw_detections:
            try:
                if self.frame_count % 30 == 0:
                    felix_count = sum(1 for det in self.raw_detections if det.get("is_felix", False))
                    self.logger.debug(f"[VISUALIZE] Frame #{self.frame_count}: Fallback to {len(self.raw_detections)} raw detections ({felix_count} Felix)")

                
                # Draw raw detections
                for det in self.raw_detections:
                    x, y, w, h = det["box"]
                    is_felix = det.get("is_felix", False)
                    confidence = det.get("confidence", 0.0)
                    
                    # Determine color (green for Felix, red for Not Felix)
                    color = (0, 255, 0) if is_felix else (0, 0, 255)  # BGR format
                    
                    # Draw box
                    cv2.rectangle(frame_copy, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label
                    label = f"Felix: {confidence:.2f}" if is_felix else f"Not Felix: {confidence:.2f}"
                    cv2.putText(
                        frame_copy,
                        label,
                        (x, max(0, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        color,
                        2
                    )
                
                return frame_copy
                
            except Exception as e:
                self.logger.error(f"[VISUALIZE] Error with fallback visualization: {e}")
                # Will fall through to default
        
        # Option 3: No detections to visualize
        else:
            if self.frame_count % 30 == 0:
                self.logger.debug(f"[VISUALIZE] Frame #{self.frame_count}: No tracks or detections to visualize")
        
        return frame_copy
        
    # Inside FelixTrackingClient class:

    async def display_loop(self):
        """Display processed frames using the shared current_frame (Non-Blocking Visualization)."""
        self.logger.info("\n=== Starting display loop ===")
        window_name = 'Felix Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        try:
            while True:
                frame_to_process = None
                # Get the current frame safely
                async with self.frame_lock: # Lock is still needed for safe access
                    if self.current_frame is not None:
                        frame_to_process = self.current_frame.copy()
                    else:
                        # If no frame yet, sleep briefly and continue
                        await asyncio.sleep(0.01)
                        continue # Skip to next loop iteration

                if frame_to_process is not None:
                    processed_frame = None
                    # --- Process visualization in a separate thread ---
                    try:
                        viz_start_time = time.time()
                        # Run the synchronous visualize_frame function in a thread
                        # Note: visualize_frame now needs access to self.tracked_detections
                        # or self.raw_detections, which were updated by receive_results
                        processed_frame = await asyncio.to_thread(self.visualize_frame, frame_to_process)
                        # ^^^ Event loop is free during visualization ^^^
                        viz_duration = time.time() - viz_start_time
                        # Log if visualization takes significant time (optional)
                        if viz_duration > 0.05:
                            self.logger.debug(f"Visualization took {viz_duration:.4f}s")

                    except Exception as viz_err:
                        self.logger.error(f"Error during threaded visualization: {viz_err}", exc_info=True)
                        # Fallback to showing the raw frame on visualization error
                        processed_frame = frame_to_process

                    # --- Display the result ---
                    if processed_frame is not None:
                        try:
                            cv2.imshow(window_name, processed_frame)
                        except Exception as display_err:
                            # Catch potential errors from cv2.imshow if window closed unexpectedly etc.
                            self.logger.error(f"Error during cv2.imshow: {display_err}")
                            break # Exit loop if display fails badly


                # --- Handle Quit Key (Still potentially slightly blocking) ---
                # Consider moving this to a separate thread if it proves problematic
                key = cv2.waitKey(1) & 0xFF # Use mask for compatibility
                if key == ord('q'):
                    self.logger.info("User pressed 'q'. Exiting display loop...")
                    # Signal other tasks to stop cleanly if possible?
                    break # Exit this loop

                # --- Control display loop rate ---
                # Adjust sleep target based on desired FPS, considering viz time is now offloaded
                await asyncio.sleep(0.1) # Aim for ~30-60 FPS display updates

        except asyncio.CancelledError:
            self.logger.info("Display loop cancelled.")
        except Exception as e:
            # Catch errors like window creation failure etc.
            self.logger.error(f"Error in display_loop: {e}", exc_info=True)
            traceback.print_exc()
        finally:
            # Ensure cleanup happens
            try:
                cv2.destroyAllWindows()
                self.logger.info("Display window destroyed.")
            except Exception as destroy_err:
                self.logger.error(f"Error destroying cv2 windows: {destroy_err}")
        
    async def run(self, video_source=0, max_retries=5, initial_retry_delay=5.0, max_retry_delay=60.0):
        """
        Run the client with automatic reconnection logic.

        Args:
            video_source: The video source index or path.
            max_retries: Maximum number of consecutive connection attempts.
            initial_retry_delay: Initial delay (seconds) between retries.
            max_retry_delay: Maximum delay (seconds) between retries (for exponential backoff).
        """
        # --- Initial Camera Test ---
        self.logger.info("\n=== Testing camera access ===")
        try:
            cap_test = cv2.VideoCapture(video_source)
            if not cap_test.isOpened():
                self.logger.error(f"FATAL: Cannot access camera source '{video_source}'. Exiting.")
                return # Exit if camera cannot be opened initially
            ret, _ = cap_test.read()
            cap_test.release()
            if not ret:
                self.logger.error(f"FATAL: Cannot read initial frame from camera source '{video_source}'. Exiting.")
                return # Exit if initial read fails
            self.logger.info("Camera test successful")
        except Exception as cam_err:
            self.logger.error(f"FATAL: Error during initial camera test: {cam_err}", exc_info=True)
            return

        # --- Reconnection Loop ---
        attempt = 0
        current_retry_delay = initial_retry_delay
        connect_timeout = 20.0 # Timeout for the connection attempt itself

        while attempt < max_retries:
            attempt += 1
            self.logger.info(f"Connection attempt {attempt}/{max_retries} to {self.server_url}...")
            websocket = None
            all_tasks = [] # Keep track of all tasks for this attempt

            try:
                # --- Attempt WebSocket Connection ---
                self.logger.debug(f"Attempting websocket.connect with {connect_timeout}s timeout...")
                websocket = await asyncio.wait_for(
                    websockets.connect(
                        self.server_url,
                        ping_interval=15,  # Interval for server to send pings
                        ping_timeout=30,   # How long server waits for pong
                        close_timeout=10,  # Timeout for close handshake
                        # Set reasonable size limits
                        max_size=10 * 1024 * 1024, # 10MB limit
                    ),
                    timeout=connect_timeout
                )
                self.logger.info(f"Connection successful! ({websocket.remote_address})")
                # Reset retry counters on success
                attempt = 0
                current_retry_delay = initial_retry_delay

                # --- Launch Core Tasks ---
                self.logger.info("Launching client tasks (capture/send, receive, display)...")
                capture_send_task = asyncio.create_task(
                    self.capture_and_send_frames(websocket, video_source), name="CaptureSendTask"
                )
                receive_task = asyncio.create_task(
                    self.receive_results(websocket), name="ReceiveTask"
                )
                display_task = asyncio.create_task(
                    self.display_loop(), name="DisplayTask"
                )
                all_tasks = [capture_send_task, receive_task, display_task]

                # --- Monitor Tasks ---
                # Wait until one task completes (normally or with error)
                self.logger.info("Monitoring client tasks...")
                done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
                self.logger.info("Monitor detected task completion or failure.")

                # --- Handle Task Completion/Failure ---
                for task in done:
                    task_name = task.get_name()
                    try:
                        exc = task.exception() # Check if task raised an exception
                        if exc:
                            # Log error and re-raise to trigger reconnect logic
                            self.logger.error(f"Task '{task_name}' failed: {exc}", exc_info=exc)
                            raise exc # Trigger the outer except block
                        else:
                            # Task finished without error (e.g., user pressed 'q' in display)
                            self.logger.info(f"Task '{task_name}' completed normally. Initiating clean shutdown.")
                            # If one task finishes cleanly, we might want to stop everything
                            # Cancel pending tasks and exit the outer loop
                            for p_task in pending:
                                p_task.cancel()
                            if pending:
                                await asyncio.wait(pending) # Wait for cancellations
                            return # Exit the run method cleanly

                    except asyncio.CancelledError:
                        self.logger.info(f"Task '{task_name}' was cancelled (likely during shutdown).")
                        # If cancellation was initiated externally, we might want to exit cleanly
                        # Re-cancel pending tasks just in case
                        for p_task in pending:
                                p_task.cancel()
                        if pending:
                                await asyncio.wait(pending)
                        return # Exit run method

            # --- Handle Connection Errors & Task Failures that Trigger Reconnect ---
            except websockets.exceptions.ConnectionClosed as e:
                self.logger.warning(f"Connection closed unexpectedly: {e}. Will retry.")
            except ConnectionRefusedError:
                self.logger.error(f"Connection refused by server at {self.server_url}.")
                self.logger.info("Server might be down or restarting.")
            except asyncio.TimeoutError:
                    self.logger.error(f"Connection attempt to {self.server_url} timed out after {connect_timeout} seconds.")
            except OSError as e:
                    self.logger.error(f"Network OS error: {e}")
            except Exception as e:
                    # Catch other exceptions from connect or task failures re-raised above
                    self.logger.error(f"Unexpected error during operation: {e}", exc_info=True)

            # --- Cleanup Before Retrying ---
            self.logger.info("Performing cleanup before next action...")
            # 1. Cancel any remaining tasks from this attempt
            for task in all_tasks: # Use the list from the current attempt scope
                if task and not task.done():
                    self.logger.debug(f"Cancelling task {task.get_name()}...")
                    task.cancel()
            # Give cancelled tasks a moment to process cancellation
            # Gather ensures we wait even if task handles cancellation quickly
            cancelled_tasks = [t for t in all_tasks if t and t.cancelled()]
            pending_tasks = [t for t in all_tasks if t and not t.done()]
            if pending_tasks:
                    self.logger.debug(f"Waiting for {len(pending_tasks)} pending tasks to cancel...")
                    await asyncio.wait(pending_tasks, timeout=2.0) # Short wait
                    # Log tasks that didn't cancel quickly
                    for task in pending_tasks:
                        if not task.done():
                            self.logger.warning(f"Task {task.get_name()} did not finish cancelling quickly.")


            # 2. Ensure websocket is closed (if it was ever opened)
            if websocket and not websocket.closed:
                self.logger.debug("Closing potentially open websocket connection...")
                try:
                    await websocket.close()
                except Exception as close_err:
                    self.logger.warning(f"Error closing websocket during cleanup: {close_err}")

            # --- Decide Whether to Retry ---
            if attempt < max_retries:
                self.logger.info(f"Waiting {current_retry_delay:.1f} seconds before retry #{attempt + 1}...")
                await asyncio.sleep(current_retry_delay)
                # Exponential backoff
                current_retry_delay = min(current_retry_delay * 1.5, max_retry_delay)
            else:
                self.logger.error("Maximum connection retry attempts reached. Exiting client.")
                break # Exit the while loop

        self.logger.info("FelixTrackingClient run method finished.")
# Main entry point
async def main():
    # Create a logger for the main function
    logger = logging.getLogger("IseeYou.main")
    logger.info("Starting Felix Tracking Client...")
    client = FelixTrackingClient(server_url="ws://localhost:8080")
    await client.run(video_source=0)  # Use webcam

if __name__ == "__main__":
    # Setup logging for the main script
    logging.basicConfig(
        filename='isee_you.log',
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("IseeYou")
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nProgram interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        traceback.print_exc()