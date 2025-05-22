##IseeYouClass

import asyncio
import cv2
import websockets
import logging
import queue
import threading
import time
import random  # For jitter in reconnection attempts

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger('felix_client')

class FelixTrackingClient:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.queue = queue.Queue(maxsize=10)  # Limit queue size
        self.running = True
        
        # Start camera thread
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _capture_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                # If queue full, remove oldest frame
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                # Add new frame
                self.queue.put(frame)
            time.sleep(0.1)  # Increased sleep to reduce CPU usage and frame rate
    
    def get_frame(self):
        return self.queue.get(block=True, timeout=1.0)
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

async def main():
    # Configuration
    server_uri = "ws://localhost:8080"
    camera = FelixTrackingClient(0)
    target_fps = 5  # Reduced to 5 FPS for better stability
    max_reconnect_attempts = 10
    
    try:
        # Main reconnection loop
        reconnect_attempt = 0
        while True:
            try:
                if reconnect_attempt > 0:
                    # Calculate backoff with jitter to prevent reconnection storms
                    backoff_seconds = min(30, 2 ** reconnect_attempt) + random.uniform(0, 1)
                    logger.info(f"Reconnection attempt {reconnect_attempt} in {backoff_seconds:.2f} seconds...")
                    await asyncio.sleep(backoff_seconds)
                
                logger.info(f"Connecting to {server_uri}...")
                async with websockets.connect(server_uri) as websocket:
                    logger.info("Connected to server")
                    reconnect_attempt = 0  # Reset on successful connection
                    
                    # Create both tasks with proper management
                    send_task = asyncio.create_task(send_frames(websocket, camera, target_fps))
                    ping_task = asyncio.create_task(handle_pings(websocket))
                    
                    # Wait for either task to complete
                    done, pending = await asyncio.wait(
                        [send_task, ping_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Check which task completed and why
                    for task in done:
                        if task.exception():
                            logger.error(f"Task failed with error: {task.exception()}")
                    
                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            logger.debug(f"Task {task.get_name()} cancelled")
                    
                    # If we get here, a task completed - try to reconnect
                    logger.info("Connection ended, will try to reconnect...")
                    
            except (websockets.exceptions.WebSocketException, 
                    ConnectionRefusedError, 
                    ConnectionResetError,
                    ConnectionError,
                    OSError) as e:
                reconnect_attempt += 1
                logger.error(f"Connection error: {e}")
                
                if reconnect_attempt > max_reconnect_attempts:
                    logger.error(f"Maximum reconnection attempts ({max_reconnect_attempts}) reached. Giving up.")
                    break
                    
    except KeyboardInterrupt:
        logger.info("Program interrupted by user")
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
    finally:
        camera.release()
        logger.info("Disconnected from server")

async def send_frames(websocket, camera, target_fps):
    """Send frames with yields to the event loop"""
    frame_interval = 1.0 / target_fps
    
    while True:
        send_start = time.time()
        
        try:
            # Get frame (non-blocking via thread)
            frame = await asyncio.to_thread(camera.get_frame)
            
            # Encode frame (offload to thread)
            encode_result = await asyncio.to_thread(
                cv2.imencode, '.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30]
            )
            _, jpeg_data = encode_result
            jpeg_bytes = jpeg_data.tobytes()
            
            # KEY FIX 1: Yield control briefly before sending
            await asyncio.sleep(0.01)
            
            # Send the frame
            logger.debug(f"Sending frame: {len(jpeg_bytes)} bytes")
            await websocket.send(jpeg_bytes)
            
            # KEY FIX 2: Yield control after sending
            await asyncio.sleep(0.01)
            
            # Calculate sleep time to maintain target FPS
            elapsed = time.time() - send_start
            sleep_time = max(0, frame_interval - elapsed)
            
            # Sleep until next frame (allows other tasks to run)
            await asyncio.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Error in send_frames: {e}")
            await asyncio.sleep(1)  # Prevent tight error loop
            raise  # Re-raise to trigger reconnection

async def handle_pings(websocket):
    """Dedicated task just for handling connection control frames"""
    while True:
        try:
            # This will automatically handle PING/PONG frames
            # But having a dedicated receive task helps prioritize them
            message = await websocket.recv()
            
            # If we receive a message, it's not a PING/PONG (handled automatically)
            # Process any server messages here
            logger.debug(f"Received message from server: {message[:50]}...")
            
        except websockets.exceptions.ConnectionClosed as e:
            logger.warning(f"Connection closed: {e}")
            break
        except Exception as e:
            logger.error(f"Error in handle_pings: {e}")
            await asyncio.sleep(1)  # Prevent tight error loop
            raise  # Re-raise to trigger reconnection

if __name__ == "__main__":
    asyncio.run(main())