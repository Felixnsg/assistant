import asyncio
import websockets
import cv2
import numpy as np
import json
import time
import traceback
import logging
import os # For os.urandom if testing without camera

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - Client - %(message)s')

# Shared state for stopping
stop_event = asyncio.Event()

async def capture_and_send_minimal(websocket, video_source, target_fps=15):
    """Capture frames and send them with minimal processing."""
    logging.info(f"Starting capture/send (Target FPS: {target_fps})")
    send_interval = 1.0 / target_fps
    cap = None
    use_dummy_data = False

    try:
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            logging.error("Cannot access camera, using dummy data.")
            use_dummy_data = True
        else:
             ret, _ = cap.read()
             if not ret:
                logging.error("Cannot read from camera, using dummy data.")
                use_dummy_data = True
                cap.release()
                cap = None # Ensure cap is None

    except Exception as cam_err:
         logging.error(f"Camera init error {cam_err}, using dummy data.")
         use_dummy_data = True
         if cap: cap.release()


    frame_counter = 0
    last_send_time = time.monotonic()

    while not stop_event.is_set():
        frame = None
        if not use_dummy_data and cap and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.error("Error reading frame")
                await asyncio.sleep(0.1)
                continue
        elif use_dummy_data:
             # Send dummy data if camera fails
             frame = np.zeros((480, 640, 3), dtype=np.uint8) # Example size
             # Or just send random bytes without even creating a frame
             # frame_bytes = os.urandom(100 * 1024) # 100KB dummy
             # await websocket.send(frame_bytes) ... continue


        frame_counter += 1
        current_time = time.monotonic()

        if current_time - last_send_time >= send_interval:
             if frame is not None:
                # --- Minimal Processing: Encode fast or send raw ---
                try:
                    # Option A: Fast JPEG encode (adjust quality)
                    encode_start = time.monotonic()
                    # Use lower quality for speed/size
                    # success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 50])
                    # frame_bytes = buffer.tobytes() if success else None

                    # Option B: Send raw bytes (higher bandwidth)
                    frame_bytes = frame.tobytes() # Get raw bytes
                    # Add shape info if sending raw:
                    # shape_info = json.dumps({"shape": frame.shape, "dtype": str(frame.dtype)})
                    # await websocket.send(shape_info) # Send metadata first
                    # await websocket.send(frame_bytes) # Then raw data
                    # For this test, let's just send raw bytes of a fixed size dummy if needed
                    if use_dummy_data: frame_bytes = os.urandom(480*640*3 // 4) # Smaller dummy


                    encode_duration = time.monotonic() - encode_start
                    # logging.debug(f"Minimal processing took {encode_duration*1000:.2f} ms")

                    if frame_bytes:
                         # --- Send ---
                         try:
                             await websocket.send(frame_bytes)
                             last_send_time = current_time # Update time on successful send
                             logging.debug(f"Sent frame {frame_counter} ({len(frame_bytes)} bytes)")
                         except websockets.exceptions.ConnectionClosed:
                             logging.warning("Connection closed during send.")
                             stop_event.set() # Signal other tasks to stop
                             break
                         except Exception as send_err:
                             logging.error(f"Error sending frame {frame_counter}: {send_err}")
                             await asyncio.sleep(0.5) # Pause before retrying send loop
                    else:
                        logging.error("Minimal processing failed.")

                except Exception as e:
                     logging.error(f"Error in send processing: {e}")

        # Yield control briefly, aim for capture rate >> send rate
        await asyncio.sleep(0.01)

    if cap and cap.isOpened():
        cap.release()
    logging.info("Capture/send task finished.")


async def receive_minimal(websocket):
    """Receive simple JSON responses."""
    logging.info("Starting receiver")
    message_count = 0
    try:
        async for message in websocket:
            message_count += 1
            try:
                # Use to_thread just in case JSON is huge, but likely fast here
                data = await asyncio.to_thread(json.loads, message)
                logging.info(f"Received response #{message_count}: {data}")
            except json.JSONDecodeError:
                logging.error(f"Received non-JSON: {message[:100]}")
            except Exception as e:
                 logging.error(f"Error processing received message: {e}")

    except websockets.exceptions.ConnectionClosedOK:
        logging.info("Receiver connection closed normally.")
    except websockets.exceptions.ConnectionClosedError as e:
        logging.warning(f"Receiver connection closed with error: {e}")
    except asyncio.CancelledError:
         logging.info("Receiver task cancelled.")
    except Exception as e:
        logging.error(f"Unhandled error in receive loop: {e}", exc_info=True)
    finally:
        stop_event.set() # Signal sender to stop if receiver dies
        logging.info("Receive task finished.")

async def main_client(server_url="ws://localhost:8080", video_source=0, fps=15):
    logging.info(f"Starting minimal client for {server_url}")
    while not stop_event.is_set(): # Simple reconnect loop
        try:
            async with websockets.connect(server_url, ping_interval=15, ping_timeout=45) as websocket:
                logging.info("Connected to dummy server.")
                sender_task = asyncio.create_task(capture_and_send_minimal(websocket, video_source, target_fps=fps))
                receiver_task = asyncio.create_task(receive_minimal(websocket))
                # Wait for either to finish
                done, pending = await asyncio.wait([sender_task, receiver_task], return_when=asyncio.FIRST_COMPLETED)

                for task in pending: # Cancel the other task if one finishes
                    task.cancel()
                if pending:
                     await asyncio.wait(pending) # Wait for cancellation

                # Check if stop was signalled or if a task failed unexpectedly
                if stop_event.is_set():
                     logging.info("Stop event set, exiting.")
                     break

                # If we get here, one task finished unexpectedly without setting stop_event
                logging.warning("A task finished unexpectedly, will retry connection.")


        except (websockets.exceptions.ConnectionClosed, ConnectionRefusedError,
                OSError, asyncio.TimeoutError) as e:
            logging.error(f"Connection failed: {e}. Retrying in 5s...")
        except Exception as e:
             logging.error(f"Main client loop error: {e}. Retrying in 5s...")

        if not stop_event.is_set(): # Only sleep if not stopping
             await asyncio.sleep(5) # Wait before retrying connection
        else:
             break # Exit loop if stopping

if __name__ == "__main__":
    # Example usage: Send 15 FPS to dummy server
    try:
        # Use Ctrl+C to stop
        asyncio.run(main_client(fps=15))
    except KeyboardInterrupt:
        logging.info("Client stopped by user.")
        stop_event.set()