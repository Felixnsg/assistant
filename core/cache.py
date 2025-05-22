import asyncio
import cv2
import websockets
import logging
import queue
import threading
import time
import json
from pathlib import Path
import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('simple_felix_client')

class ThreadedCamera:
    # This class remains unchanged
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

class VisualContextCache:
    """Simplified cache for visual detection data that's compatible with the chat system"""
    
    def __init__(self):
        # Thread safety lock
        self.lock = asyncio.Lock()
        
        # Detection data structure (matching the format expected by utilities.py)
        self.detection_data = {
            "is_felix": False,      # Whether Felix is currently detected
            "confidence": 0.0,      # Confidence level (0.0 to 1.0)
            "timestamp": time.time(), # When this data was last updated
            "data_available": False  # Whether vision system is providing data
        }
        
        # Create directory for cache dumps
        self.cache_dir = Path("cache_dumps")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Track when we last wrote to file
        self.last_file_write = 0
        self.file_write_interval = 2.0  # Write to file every 2 seconds
        
        # Additional debug tracking
        self.update_count = 0
        self.detection_history = []  # Limited history of detections
        
        self.logger = logging.getLogger("SimpleVisualContextCache")
        self.logger.info("Simple Visual Context Cache initialized")
    
    async def update(self, detections):
        """Updates the cache with latest detection data"""
        try:
            if not detections:
                return False
                
            async with self.lock:
                # Find best Felix detection
                best_confidence = 0.0
                felix_detected = False
                
                for detection in detections:
                    confidence = detection.get('confidence', 0.0)
                    is_felix = detection.get('is_felix', False)
                    
                    if is_felix and confidence > best_confidence:
                        best_confidence = confidence
                        felix_detected = True
                
                # Update detection data
                self.detection_data.update({
                    "is_felix": felix_detected,
                    "confidence": best_confidence,
                    "timestamp": time.time(),
                    "data_available": True
                })
                
                # Add to detection history (keep last 10)
                self.update_count += 1
                self.detection_history.append({
                    "update_number": self.update_count,
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "is_felix": felix_detected,
                    "confidence": best_confidence,
                    "raw_detections": detections
                })
                if len(self.detection_history) > 10:
                    self.detection_history.pop(0)
                
                # Check if it's time to write to file
                current_time = time.time()
                if current_time - self.last_file_write >= self.file_write_interval:
                    await self._write_cache_to_file()
                    self.last_file_write = current_time
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating detection data: {e}")
            return False
    
    async def _write_cache_to_file(self):
        """Write current cache state to a JSON file"""
        try:
            # Create a comprehensive debug dump
            debug_data = {
                "current_state": self.detection_data,
                "update_count": self.update_count,
                "detection_history": self.detection_history,
                "dump_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Create filename with timestamp
            filename = self.cache_dir / f"cache_dump_{int(time.time())}.json"
            
            # Write to file
            with open(filename, 'w') as f:
                json.dump(debug_data, f, indent=2)
            
            # Also write to a "latest.json" file that's always overwritten
            latest_file = self.cache_dir / "latest.json"
            with open(latest_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
                
            self.logger.info(f"Cache data written to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error writing cache to file: {e}")
            return False
    
    async def get_detection_info(self):
        """Get current detection info thread-safely"""
        async with self.lock:
            return self.detection_data.copy()
    
    async def is_data_stale(self, max_age_seconds=5.0):
        """Check if data is stale"""
        async with self.lock:
            return (time.time() - self.detection_data['timestamp']) > max_age_seconds
    
    # Compatibility method for utilities.py
    async def _update_loop(self, client, update_interval=0.5):
        """Compatibility method that updates from client periodically"""
        self.logger.info(f"Visual context update loop started (interval: {update_interval}s)")
        while True:
            try:
                # Check if client has raw_detections attribute and data
                if hasattr(client, "raw_detections") and client.raw_detections:
                    await self.update(client.raw_detections)
                await asyncio.sleep(update_interval)
            except asyncio.CancelledError:
                self.logger.info("Update loop cancelled")
                break
            except Exception as e:
                self.logger.error(f"Error in visual context update loop: {e}")
                await asyncio.sleep(update_interval * 2)
    
    async def share_info_AI(self, chat_manager):
        """Share vision detection information with the AI assistant (compatible with existing system)"""
        self.logger.info("Sharing vision context with AI assistant")
        prompt = "[]"
        try:
            # Get the latest detection info
            detection_info = await self.get_detection_info()
            
            # Check if data is stale
            if await self.is_data_stale():
                prompt = "[Visual context: Vision system data is stale or unavailable]"
                self.logger.warning("Using stale detection data for AI communication")
            
            elif detection_info and detection_info.get("data_available", False):
                if detection_info.get("is_felix", False):
                    prompt = (
                        f"[Visual context: Felix has been detected with "
                        f"{detection_info['confidence']*100:.1f}% confidence.]"
                    )
                else:
                    prompt = "[Visual context: No Felix detected in camera view.]"
            else:
                prompt = "[Visual context: Vision system is not providing data]"
                        
            # Send the prompt to the LLM
            self.logger.info(f"Sending vision context to LLM: {prompt}")
            if not chat_manager:
                self.logger.error("Cannot share info with AI: chat_manager object is None.")
                return None
            
            context_response = await chat_manager._call_llm(prompt)
            if context_response and not context_response.startswith("Blocked:"):
                return {"context_prompt": prompt, "context_response": context_response}
            else:
                self.logger.error(f"LLM call for visual context failed or was blocked: {context_response}")
                return None
            
        except Exception as e:
            self.logger.error(f"Error sharing vision context with AI: {e}")
            return None

# Rest of the file remains unchanged
async def main():
    """Main function that integrates camera with cache"""
    # Configuration
    server_uri = "ws://localhost:8080"
    camera = ThreadedCamera(0)
    target_fps = 30
    visual_cache = VisualContextCache()
    
    # Track latest frame and detections
    current_frame = None
    raw_detections = []
    
    try:
        async with websockets.connect(server_uri) as websocket:
            logger.info("Connected to server")
            
            while True:
                # PART 1: Send frames
                try:
                    # Get and prepare frame
                    frame = await asyncio.to_thread(camera.get_frame)
                    current_frame = frame.copy()
                    
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
                            
                            # Store the detections for reference and update cache
                            if latest_detections:
                                raw_detections = latest_detections
                                # Update the cache with new detections
                                await visual_cache.update(latest_detections)
                                
                        except asyncio.TimeoutError:
                            break
                        except Exception as e:
                            logger.error(f"Error receiving message: {e}")
                
                except Exception as e:
                    logger.error(f"Error processing detections: {e}")
                    await asyncio.sleep(0.1)
                
    except Exception as e:
        logger.error(f"Connection error: {e}")
    finally:
        camera.release()
        logger.info("Disconnected from server")

def process_message(message):
    """Convert websocket message to detection data"""
    try:
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