import asyncio
import websockets
import cv2
import numpy as np
import json
import time
import traceback
import logging

# NOTE: Removed 'import supervision as sv' - visualization happens server-side

class FelixTrackingClient:
    """
    Lightweight client: Captures video, sends low-quality frames,
    receives annotated frames, and displays them.
    """

    def __init__(self, server_url="ws://localhost:8080"):
        self.logger = logging.getLogger("IseeYou.FelixTrackingClient")
        self.logger.info("\n=== Initializing Lightweight FelixTrackingClient ===")
        self.server_url = server_url

        # For receiving and displaying annotated frames
        self.latest_annotated_frame = None
        self.annotated_frame_lock = asyncio.Lock()

        self.logger.info("Lightweight FelixTrackingClient initialized successfully")

    async def capture_and_send_frames(self, websocket, video_source, target_fps=10, jpeg_quality=50):
        """Capture frames, encode LOW-QUALITY JPEG non-blocking, and send."""
        self.logger = logging.getLogger("IseeYou.FelixTrackingClient.CaptureSend")
        self.logger.info(f"\n=== Starting video capture (Target Send FPS: {target_fps}, Quality: {jpeg_quality}) ===")

        if target_fps <= 0: target_fps = 1 # Avoid division by zero
        send_interval = 1.0 / target_fps
        self.logger.info(f"Calculated send interval: {send_interval:.3f} seconds")

        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            self.logger.error("Error: Cannot access camera")
            return # Exit if camera cannot be opened

        try:
            frame_counter = 0
            last_send_time = time.monotonic()

            while True: # Main loop runs fast to provide frames for potential local preview if added later
                ret, frame = cap.read()
                if not ret:
                    self.logger.error("Error: Cannot read frame from camera")
                    await asyncio.sleep(0.1)
                    continue

                # --- Check if it's time to send the next frame ---
                current_time = time.monotonic()
                if current_time - last_send_time >= send_interval:
                    frame_counter += 1
                    self.logger.debug(f"Preparing frame {frame_counter} for sending...")

                    # --- Encode as low-quality JPEG in thread ---
                    try:
                        encode_start_time = time.monotonic()
                        params = [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
                        encoded_result = await asyncio.to_thread(cv2.imencode, ".jpg", frame, params)
                        success_flag, encoded_buffer = encoded_result
                        encode_duration = time.monotonic() - encode_start_time
                        self.logger.debug(f"Encoding frame {frame_counter} finished in {encode_duration:.4f} seconds")

                    except Exception as encode_error:
                        self.logger.error(f"Error during threaded cv2.imencode for frame {frame_counter}: {encode_error}", exc_info=True)
                        success_flag = False
                        encoded_buffer = None

                    if not success_flag or encoded_buffer is None:
                        self.logger.error(f"Error: cv2.imencode failed for frame {frame_counter}")
                        # Don't update last_send_time, try again next eligible cycle
                    else:
                        frame_bytes = encoded_buffer.tobytes()

                        # --- Send the encoded frame ---
                        try:
                            send_start_time = time.monotonic()
                            await websocket.send(frame_bytes)
                            send_duration = time.monotonic() - send_start_time
                            # Update last_send_time ONLY after successful send attempt
                            last_send_time = current_time
                            self.logger.debug(f"Sent frame {frame_counter} ({len(frame_bytes)} bytes, send took {send_duration:.4f}s)")

                        except websockets.exceptions.ConnectionClosed:
                            self.logger.warning("Connection closed during send. Exiting capture loop.")
                            break # Exit the loop
                        except Exception as send_error:
                            self.logger.error(f"Error sending frame {frame_counter}: {send_error}", exc_info=True)
                            # Don't update last_send_time on send error, will retry sending next cycle
                            await asyncio.sleep(0.5) # Pause briefly after send error

                # --- Yield control briefly - aims for smooth capture rate ---
                await asyncio.sleep(0.01) # Adjust if needed based on camera speed

        except asyncio.CancelledError:
            self.logger.info("Capture and send task cancelled.")
        except Exception as e:
            self.logger.error(f"Unhandled error in capture_and_send_frames loop: {e}", exc_info=True)
        finally:
            if cap and cap.isOpened(): # Check if cap exists and is open
                cap.release()
            self.logger.info("Camera released.")


    async def receive_annotated_frames(self, websocket):
        """Receive annotated frame bytes from server and decode NON-BLOCKINGLY."""
        self.logger = logging.getLogger("IseeYou.FelixTrackingClient.Receive")
        self.logger.info("\n=== Starting to receive annotated frames ===")
        frame_counter = 0
        try:
            async for message in websocket: # Expecting binary frame data now
                frame_counter += 1
                if not isinstance(message, bytes):
                    self.logger.warning(f"Received non-bytes message type: {type(message)}. Skipping.")
                    continue

                self.logger.debug(f"[RECEIVER] Received annotated frame #{frame_counter} ({len(message)} bytes)")
                try:
                    # --- Decode received JPEG bytes in thread ---
                    decode_start = time.monotonic()
                    frame_bytes_np = np.frombuffer(message, dtype=np.uint8)
                    annotated_frame = await asyncio.to_thread(cv2.imdecode, frame_bytes_np, cv2.IMREAD_COLOR)
                    decode_duration = time.monotonic() - decode_start
                    self.logger.debug(f"Decode took {decode_duration:.4f}s")

                    if annotated_frame is None:
                        self.logger.error(f"[RECEIVER] Failed to decode annotated frame #{frame_counter}")
                        continue

                    # --- Store the decoded frame for display ---
                    async with self.annotated_frame_lock:
                        self.latest_annotated_frame = annotated_frame

                except Exception as e:
                    self.logger.error(f"[RECEIVER] Error decoding/storing frame #{frame_counter}: {e}", exc_info=True)

        except websockets.exceptions.ConnectionClosedOK:
            self.logger.info("[RECEIVER] Connection closed normally.")
        except websockets.exceptions.ConnectionClosedError as e:
            self.logger.warning(f"[RECEIVER] Connection closed with error: {e}")
        except asyncio.CancelledError:
            self.logger.info("[RECEIVER] Receive task cancelled.")
        except Exception as e:
            self.logger.error(f"[RECEIVER] Unhandled error in receive loop: {e}", exc_info=True)


    # update_tracking REMOVED
    # visualize_frame REMOVED

    async def display_loop(self):
        """Displays the latest ANNOTATED frame received from the server."""
        self.logger = logging.getLogger("IseeYou.FelixTrackingClient.Display")
        self.logger.info("\n=== Starting display loop ===")
        window_name = 'Felix Tracking (Annotated)'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        last_display_time = time.monotonic()
        display_interval = 1.0 / 30.0 # Target ~30 FPS display rate

        try:
            while True:
                frame_to_display = None
                # Get the latest annotated frame safely
                async with self.annotated_frame_lock:
                    if self.latest_annotated_frame is not None:
                        # Make a copy to avoid holding lock during imshow
                        frame_to_display = self.latest_annotated_frame.copy()

                if frame_to_display is not None:
                    try:
                        cv2.imshow(window_name, frame_to_display)
                    except Exception as display_err:
                        self.logger.error(f"Error during cv2.imshow: {display_err}")
                        break # Exit loop if display fails badly
                # else:
                #     self.logger.debug("Display loop: No annotated frame available yet.")

                # --- Handle Quit Key ---
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.logger.info("User pressed 'q'. Exiting display loop...")
                    break # Exit this loop

                # --- Control display loop rate ---
                # Calculate sleep time to maintain target FPS
                current_time = time.monotonic()
                elapsed = current_time - last_display_time
                sleep_time = max(0, display_interval - elapsed)
                await asyncio.sleep(sleep_time)
                last_display_time = time.monotonic() # Update time after sleep

        except asyncio.CancelledError:
            self.logger.info("Display loop cancelled.")
        except Exception as e:
            self.logger.error(f"Error in display_loop: {e}", exc_info=True)
        finally:
            try:
                cv2.destroyAllWindows()
                self.logger.info("Display window destroyed.")
            except Exception as destroy_err:
                self.logger.error(f"Error destroying cv2 windows: {destroy_err}")

    # --- Run method needs minor update for task names ---
    async def run(self, video_source=0, target_send_fps=10,
                 max_retries=5, initial_retry_delay=5.0, max_retry_delay=60.0):
        """Run the lightweight client with automatic reconnection."""
        # ... (Camera test logic remains the same) ...
        self.logger.info("Camera test successful")

        attempt = 0
        current_retry_delay = initial_retry_delay
        connect_timeout = 20.0

        while attempt < max_retries:
            attempt += 1
            self.logger.info(f"Connection attempt {attempt}/{max_retries} to {self.server_url}...")
            websocket = None
            all_tasks = []

            try:
                try:
                    websocket = await asyncio.wait_for(
                        websockets.connect(
                            self.server_url,
                            ping_interval=15,
                            ping_timeout=45,
                            close_timeout=10,
                            max_size=10 * 1024 * 1024,
                        ),
                        timeout=connect_timeout
                    )
                    self.logger.info(f"Connection successful! ({websocket.remote_address})")
                    attempt = 0
                    current_retry_delay = initial_retry_delay

                    self.logger.info("Launching client tasks (capture/send, receive, display)...")
                    capture_send_task = asyncio.create_task(
                        self.capture_and_send_frames(websocket, video_source, target_fps=target_send_fps),
                        name="CaptureSendTask"
                    )
                    # Renamed receive function
                    receive_annotated_task = asyncio.create_task(
                        self.receive_annotated_frames(websocket), name="ReceiveAnnotatedTask"
                    )
                    display_task = asyncio.create_task(
                        self.display_loop(), name="DisplayTask"
                    )
                    # Updated task list
                    all_tasks = [capture_send_task, receive_annotated_task, display_task]

                    self.logger.info("Monitoring client tasks...")
                    done, pending = await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
                    self.logger.info("Monitor detected task completion or failure.")

                    # --- Handle Task Completion/Failure (LOGIC REMAINS THE SAME) ---
                    should_exit_cleanly = False
                    should_trigger_reconnect = False
                    for task in done:
                        task_name = task.get_name()
                        try:
                            exc = task.exception()
                            if exc:
                                self.logger.error(f"Task '{task_name}' failed: {exc}", exc_info=exc)
                                should_trigger_reconnect = True
                            else:
                                self.logger.info(f"Task '{task_name}' completed normally.")
                                if task_name == "DisplayTask":
                                    self.logger.info("DisplayTask finished normally. Initiating clean shutdown.")
                                    should_exit_cleanly = True
                                else: # Capture or Receive finished normally -> connection issue
                                    self.logger.warning(f"Task '{task_name}' finished normally, likely connection loss. Will retry.")
                                    should_trigger_reconnect = True
                        except asyncio.CancelledError:
                            self.logger.info(f"Task '{task_name}' was cancelled.")
                            should_exit_cleanly = True
                        if should_exit_cleanly or should_trigger_reconnect: break

                    if should_exit_cleanly:
                        self.logger.info("Clean exit condition met. Shutting down remaining tasks.")
                        for p_task in pending:
                            if p_task and not p_task.done(): p_task.cancel()
                        if pending: await asyncio.wait(pending)
                        return # EXIT run method cleanly

                    if should_trigger_reconnect:
                        self.logger.info("Reconnect condition met. Proceeding to cleanup and retry.")
                        pass # Continue to cleanup outside this block

                    elif not pending:
                         self.logger.warning("All tasks finished, but no explicit exit or reconnect condition met.")

                except Exception as e:
                     # Re-raise inner exceptions to be caught by the outer block for reconnect
                     raise e

            except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError,
                    asyncio.TimeoutError, OSError, Exception) as e:
                 # (Logging for different error types remains the same)
                 if isinstance(e, websockets.exceptions.ConnectionClosed): self.logger.warning(f"Connection closed unexpectedly: {e}. Will retry.")
                 # ... other error logging ...
                 else: self.logger.error(f"An operation failed triggering reconnect: {e}", exc_info=True)

            # --- Cleanup Before Retrying OR Exiting Loop ---
            # (Cleanup logic remains the same)
            self.logger.info("Performing cleanup...")
            # ... cancel pending tasks ...
            # ... close websocket ...

            # --- Decide Whether to Retry ---
            # (Retry logic remains the same)
            if attempt < max_retries:
                 # ... log waiting, sleep, exponential backoff ...
                 await asyncio.sleep(current_retry_delay) # Ensure sleep happens before next attempt
                 current_retry_delay = min(current_retry_delay * 1.5, max_retry_delay)
            else:
                self.logger.error("Maximum connection retry attempts reached. Exiting client.")
                break # Exit the while loop
        # --- End of While Loop ---

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