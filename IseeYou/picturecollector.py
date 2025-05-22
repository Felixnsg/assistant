import asyncio
import cv2
import websockets
import logging
import queue
import threading
import time
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('simple_felix_client')

class ThreadedCamera:
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        self.queue = queue.Queue(maxsize=10)
        self.running = True
        
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
        
    def _capture_loop(self):
        while self.running:
            success, frame = self.cap.read()
            if success:
                if self.queue.full():
                    try:
                        self.queue.get_nowait()
                    except queue.Empty:
                        pass
                self.queue.put(frame)
            time.sleep(0.1)
    
    def get_frame(self):
        return self.queue.get(block=True, timeout=1.0)
    
    def release(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

class SimplePictureSaver:
    def __init__(self):
        # Create folder for saved images
        self.save_dir = Path("Saved_Images")
        self.save_dir.mkdir(exist_ok=True)
        
        # Track saving
        self.last_save = 0
        self.counter = 0
        self.max_images = 1000
        self.save_interval = 3  # seconds
    
    def save_if_needed(self, frame, detection):
        """Save image if it meets our criteria and timing"""
        confidence = detection.get('confidence', 0)
        is_felix = detection.get('is_felix', False)
        
        # Only save low confidence or non-felix detections
        if confidence < 0.52 or not is_felix:
            current_time = time.time()
            if (current_time - self.last_save >= self.save_interval and 
                    self.counter < self.max_images):
                
                # Update tracking variables
                self.counter += 1
                self.last_save = current_time
                
                # Create filename and save
                filename = f"Saved_Images/detection_{int(current_time)}_{self.counter}.jpg"
                cv2.imwrite(filename, frame)
                logger.info(f"Saved image: {filename}")

async def main():
    # Configuration
    server_uri = "ws://localhost:8080"
    camera = ThreadedCamera(0)
    target_fps = 30
    picture_saver = SimplePictureSaver()
    
    # Track latest frame and detections
    current_frame = None
    latest_detections = []
    
    try:
        async with websockets.connect(server_uri) as websocket:
            logger.info("Connected to server")
            
            while True:
                # PART 1: Send frames
                try:
                    # Get and prepare frame
                    frame = await asyncio.to_thread(camera.get_frame)
                    current_frame = frame.copy()  # Save for detection processing
                    
                    # Compress and send
                    _, jpeg_data = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 30])
                    await websocket.send(jpeg_data.tobytes())
                    
                    # Wait for next frame time
                    await asyncio.sleep(1.0/target_fps)
                except Exception as e:
                    logger.error(f"Error sending frame: {e}")
                
                # PART 2: Check for incoming detection results
                try:
                    # Non-blocking check for messages
                    for _ in range(5):  # Try a few times to get any available messages
                        try:
                            message = await asyncio.wait_for(websocket.recv(), 0.01)
                            # Process message (assumed to be detections)
                            latest_detections = process_message(message)
                        except asyncio.TimeoutError:
                            break
                        except Exception as e:
                            logger.error(f"Error receiving message: {e}")
                            
                    # Process any detections with the current frame
                    if current_frame is not None and latest_detections:
                        for detection in latest_detections:
                            picture_saver.save_if_needed(current_frame, detection)
                
                except Exception as e:
                    logger.error(f"Error processing detections: {e}")
                    await asyncio.sleep(0.1)  # Prevent tight error loop
                
    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        camera.release()
        logger.info("Disconnected from server")

def process_message(message):
    """Convert websocket message to detection data"""
    # Replace this with your actual message parsing logic
    # This is just a placeholder
    try:
        # Assume message contains JSON data about detections
        import json
        return json.loads(message)
    except:
        logger.warning("Could not parse message as detection data")
        return []

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unhandled exception: {e}")