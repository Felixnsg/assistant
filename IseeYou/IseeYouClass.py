import asyncio
import cv2
import websockets
import logging
import json
import time
import numpy as np
from typing import Optional, Dict, Any
import queue
import threading

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('felix_client_v2')

class FrameCapture:
    """Efficient frame capture with automatic frame dropping"""
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        # Single frame buffer - always get the latest
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.running = True
        
        self.thread = threading.Thread(target=self._capture_loop)
        self.thread.daemon = True
        self.thread.start()
    
    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.frame_lock:
                    self.latest_frame = frame
            time.sleep(0.033)  # ~30fps capture rate
    
    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class FlowControlledClient:
    def __init__(self, server_uri: str, target_fps: int = 10):
        self.server_uri = server_uri
        self.target_fps = target_fps
        self.camera = FrameCapture()
        
        # Flow control
        self.pending_frames = 0
        self.max_pending = 3  # Max frames in flight
        self.frame_id = 0
        self.results_cache = {}
        
        # Performance metrics
        self.last_fps_time = time.time()
        self.frames_sent = 0
        self.frames_processed = 0
    
    async def run(self):
        """Main client loop with automatic reconnection"""
        reconnect_delay = 1
        
        while True:
            try:
                logger.info(f"Connecting to {self.server_uri}...")
                
                # Configure WebSocket with appropriate settings
                async with websockets.connect(
                    self.server_uri,
                    max_size=10 * 1024 * 1024,  # 10MB max message
                    ping_interval=10,
                    ping_timeout=20,
                    close_timeout=10
                ) as websocket:
                    logger.info("Connected successfully")
                    reconnect_delay = 1  # Reset delay on successful connection
                    
                    # Run send and receive tasks concurrently
                    send_task = asyncio.create_task(self._send_frames(websocket))
                    recv_task = asyncio.create_task(self._receive_results(websocket))
                    stats_task = asyncio.create_task(self._print_stats())
                    
                    # Wait for any task to complete (usually due to disconnection)
                    done, pending = await asyncio.wait(
                        [send_task, recv_task, stats_task],
                        return_when=asyncio.FIRST_COMPLETED
                    )
                    
                    # Cancel remaining tasks
                    for task in pending:
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass
                    
                    # Check what caused the completion
                    for task in done:
                        if task.exception():
                            logger.error(f"Task failed: {task.exception()}")
            
            except Exception as e:
                logger.error(f"Connection failed: {e}")
            
            # Exponential backoff for reconnection
            logger.info(f"Reconnecting in {reconnect_delay} seconds...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)
    
    async def _send_frames(self, websocket):
        """Send frames with flow control"""
        frame_interval = 1.0 / self.target_fps
        
        while True:
            try:
                # Flow control: wait if too many frames are pending
                if self.pending_frames >= self.max_pending:
                    await asyncio.sleep(0.1)
                    continue
                
                # Get latest frame
                frame = await asyncio.to_thread(self.camera.get_latest_frame)
                if frame is None:
                    await asyncio.sleep(0.1)
                    continue
                
                # Encode frame with dynamic quality based on pending frames
                quality = 30 if self.pending_frames < 2 else 20
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
                _, buffer = await asyncio.to_thread(cv2.imencode, '.jpg', frame, encode_params)
                
                # Prepare message with metadata
                self.frame_id += 1
                message = {
                    'type': 'frame',
                    'id': self.frame_id,
                    'timestamp': time.time(),
                    'data': buffer.tobytes().hex()  # Convert to hex for JSON
                }
                
                # Send frame
                await websocket.send(json.dumps(message))
                self.pending_frames += 1
                self.frames_sent += 1
                
                # Maintain target FPS
                await asyncio.sleep(frame_interval)
                
            except Exception as e:
                logger.error(f"Error sending frame: {e}")
                raise
    
    async def _receive_results(self, websocket):
        """Receive and process results"""
        while True:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                # Handle different message types
                if data.get('type') == 'result':
                    frame_id = data.get('frame_id')
                    if frame_id:
                        self.pending_frames = max(0, self.pending_frames - 1)
                        self.frames_processed += 1
                        self.results_cache[frame_id] = data.get('detections', [])
                        
                        # Clean old results
                        if len(self.results_cache) > 100:
                            oldest = min(self.results_cache.keys())
                            del self.results_cache[oldest]
                
                elif data.get('type') == 'ping':
                    # Respond to custom ping immediately
                    await websocket.send(json.dumps({'type': 'pong'}))
                    
            except json.JSONDecodeError:
                logger.error("Failed to decode message")
            except Exception as e:
                logger.error(f"Error receiving results: {e}")
                raise
    
    async def _print_stats(self):
        """Print performance statistics"""
        while True:
            await asyncio.sleep(5)
            
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            
            if elapsed > 0:
                send_fps = self.frames_sent / elapsed
                process_fps = self.frames_processed / elapsed
                
                logger.info(f"Stats - Send FPS: {send_fps:.1f}, Process FPS: {process_fps:.1f}, "
                           f"Pending: {self.pending_frames}, Cache: {len(self.results_cache)}")
                
                self.last_fps_time = current_time
                self.frames_sent = 0
                self.frames_processed = 0

async def main():
    client = FlowControlledClient(
        server_uri = "ws://localhost:8080",
        target_fps=10
    )
    
    try:
        await client.run()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        client.camera.release()

if __name__ == "__main__":
    asyncio.run(main())