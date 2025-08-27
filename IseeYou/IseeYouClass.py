import sys, os
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


import asyncio
import cv2
import websockets
import logging
import json
import time
import numpy as np
from typing import Optional, Dict, Any, Callable, List
import queue
import threading
import supervision as sv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('felix_client_v2')

class FrameCapture:
    """Efficient frame capture with automatic frame dropping"""
    def __init__(self, camera_id=0):
        self.cap = cv2.VideoCapture(camera_id)
        # Optimize camera settings for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
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
            time.sleep(0.016)  # ~60fps capture rate
    
    def get_latest_frame(self):
        with self.frame_lock:
            return self.latest_frame.copy() if self.latest_frame is not None else None
    
    def release(self):
        self.running = False
        self.thread.join()
        self.cap.release()

class FlowControlledClient:
    def __init__(self, server_uri: str, target_fps: int = 30, cache_callback: Optional[Callable] = None):
        """
        Initialize the client with optional cache callback.
        
        Args:
            server_uri: WebSocket server URI
            target_fps: Target frames per second
            cache_callback: Optional async callback function that receives detection updates
        """
        self.server_uri = server_uri
        self.target_fps = target_fps
        self.camera = FrameCapture()
        
        # Cache callback for external updates
        self.cache_callback = cache_callback
        
        # Flow control
        self.pending_frames = 0
        self.max_pending = 10
        self.frame_id = 0
        self.results_cache = {}
        
        # Frame visualization
        self.display_enabled = True
        self.latest_results = {}
        
        # Performance metrics
        self.last_fps_time = time.time()
        self.frames_sent = 0
        self.frames_processed = 0
        self.rtt_history = []
        
        # Adaptive quality
        self.current_quality = 50
        self.min_quality = 20
        self.max_quality = 80
        
        # Control flags
        self.running = False
        self.websocket = None
        self.tasks = []
        
        # Initialize Supervision components
        self._init_supervision()
    
    def _init_supervision(self):
        # ByteTrack tracker for persistent IDs
        self.tracker = sv.ByteTrack(
            track_activation_threshold=0.25,
            lost_track_buffer=30,
            minimum_matching_threshold=0.8,
            frame_rate=30
        )

        # box-only annotator (no text args here)
        self.box_annotator = sv.BoundingBoxAnnotator(thickness=2)

        # labels separately
        self.label_annotator = sv.LabelAnnotator(
            text_thickness=1,
            text_scale=0.5,
            text_position=sv.Position.TOP_LEFT
        )

        self.trace_annotator = sv.TraceAnnotator(
            thickness=2,
            trace_length=30,
            position=sv.Position.CENTER
        )
        self.fps_monitor = sv.FPSMonitor()
        self.color_palette = sv.ColorPalette.DEFAULT
        logger.info("Supervision components initialized")
    
    async def start(self):
        """Start the client - can be called from external code"""
        if self.running:
            logger.warning("Client already running")
            return True
        
        self.running = True
        try:
            await self.run()
            return True
        except Exception as e:
            logger.error(f"Error starting client: {e}")
            self.running = False
            return False
    
    async def stop(self):
        """Stop the client gracefully"""
        if not self.running:
            logger.info("Client not running")
            return True
        
        logger.info("Stopping client...")
        self.running = False
        
        # Cancel all tasks
        for task in self.tasks:
            if not task.done():
                task.cancel()
        
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
        
        # Close WebSocket
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
        
        # Release camera
        self.camera.release()
        
        # Destroy OpenCV windows
        cv2.destroyAllWindows()
        
        logger.info("Client stopped")
        return True
    
    def _detections_from_results(self, detections_data, frame_shape):
        """Convert server results to Supervision Detections format"""
        if not detections_data:
            return sv.Detections.empty()
        
        # Prepare arrays for Supervision
        xyxy = []
        confidence = []
        class_id = []
        tracker_id = []
        
        for detection in detections_data:
            # Get box coordinates
            box = detection['box']
            
            # Handle different box formats
            if len(box) == 4:
                # Could be [x1, y1, x2, y2] or [x, y, w, h]
                # Check if it's likely width/height format
                if box[2] < frame_shape[1] and box[3] < frame_shape[0]:
                    # Likely [x, y, w, h] - convert to xyxy
                    x, y, w, h = box
                    xyxy.append([x, y, x + w, y + h])
                else:
                    # Likely already [x1, y1, x2, y2]
                    xyxy.append(box)
            
            # Get confidence
            conf = detection.get('confidence', detection.get('detection_confidence', 1.0))
            confidence.append(conf)
            
            # Class ID: 0 for non-Felix, 1 for Felix
            class_id.append(1 if detection.get('is_felix', False) else 0)
            
            # Tracker ID if available
            if 'tracker_id' in detection:
                tracker_id.append(detection['tracker_id'])
        
        # Create Supervision Detections object
        if xyxy:
            detections = sv.Detections(
                xyxy=np.array(xyxy, dtype=np.float32),
                confidence=np.array(confidence, dtype=np.float32),
                class_id=np.array(class_id, dtype=np.int32),
                tracker_id=np.array(tracker_id, dtype=np.int32) if tracker_id else None
            )
            
            # Apply ByteTrack for persistent IDs
            detections = self.tracker.update_with_detections(detections)
            
            return detections
        
        return sv.Detections.empty()
    
    def _create_labels(self, detections):
        """Create labels for each detection"""
        labels = []
        
        for i in range(len(detections)):
            # Get detection info
            is_felix = detections.class_id[i] == 1
            conf = detections.confidence[i]
            track_id = detections.tracker_id[i] if detections.tracker_id is not None else -1
            
            # Create label
            person_type = "Felix" if is_felix else "PERSON!"
            label = f"{person_type} #{track_id}: {conf:.2f}"
            labels.append(label)
        
        return labels
    
    async def _update_cache_with_detections(self, detections: List[Dict[str, Any]]):
        """Send detection updates to the cache callback if provided"""
        if self.cache_callback:
            try:
                # Ensure the callback is awaited if it's async
                if asyncio.iscoroutinefunction(self.cache_callback):
                    await self.cache_callback(detections)
                else:
                    self.cache_callback(detections)
            except Exception as e:
                logger.error(f"Error calling cache callback: {e}")
    
    async def run(self):
        """Main client loop with automatic reconnection"""
        reconnect_delay = 1
        
        # Start visualization in background if enabled
        if self.display_enabled:
            display_task = asyncio.create_task(self._display_loop())
            self.tasks.append(display_task)
        
        while self.running:
            try:
                logger.info(f"Connecting to {self.server_uri}...")
                
                async with websockets.connect(
                    self.server_uri,
                    max_size=10 * 1024 * 1024,
                    ping_interval=10,
                    ping_timeout=20,
                    close_timeout=10
                ) as websocket:
                    self.websocket = websocket
                    logger.info("Connected successfully")
                    reconnect_delay = 1
                    
                    # Run tasks concurrently
                    send_task = asyncio.create_task(self._send_frames(websocket))
                    recv_task = asyncio.create_task(self._receive_results(websocket))
                    stats_task = asyncio.create_task(self._print_stats())
                    
                    self.tasks.extend([send_task, recv_task, stats_task])
                    
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
            
            if not self.running:
                break
            
            # Exponential backoff for reconnection
            logger.info(f"Reconnecting in {reconnect_delay} seconds...")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 2, 30)
    
    async def _send_frames(self, websocket):
        """Send frames with adaptive quality and flow control"""
        frame_interval = 1.0 / self.target_fps
        last_send_time = time.time()
        
        while self.running:
            try:
                # Flow control
                if self.pending_frames >= self.max_pending:
                    await asyncio.sleep(0.005)
                    continue
                
                # Frame timing
                current_time = time.time()
                time_since_last = current_time - last_send_time
                if time_since_last < frame_interval * 0.8:
                    await asyncio.sleep(0.001)
                    continue
                
                # Get latest frame
                frame = await asyncio.to_thread(self.camera.get_latest_frame)
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue
                
                # Adaptive quality
                if len(self.rtt_history) > 5:
                    avg_rtt = sum(self.rtt_history[-5:]) / 5
                    if avg_rtt > 400:
                        self.current_quality = max(self.min_quality, self.current_quality - 5)
                    elif avg_rtt < 200:
                        self.current_quality = min(self.max_quality, self.current_quality + 5)
                
                # Encode frame
                encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(self.current_quality)]
                _, encoded_frame = await asyncio.to_thread(cv2.imencode, '.jpg', frame, encode_params)
                
                # Prepare message
                self.frame_id += 1
                message = {
                    'type': 'frame',
                    'id': self.frame_id,
                    'timestamp': time.time(),
                    'data': encoded_frame.tobytes().hex()
                }
                
                # Track for RTT
                self.results_cache[self.frame_id] = {
                    'sent_at': time.time(),
                    'frame': frame
                }
                
                # Send
                await websocket.send(json.dumps(message))
                self.pending_frames += 1
                self.frames_sent += 1
                last_send_time = current_time
                
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logger.error(f"Error sending frame: {e}")
                break
    
    async def _receive_results(self, websocket):
        """Receive and process results"""
        while self.running:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get('type') == 'result':
                    frame_id = data.get('frame_id')
                    if frame_id and frame_id in self.results_cache:
                        # Calculate RTT
                        rtt = (time.time() - self.results_cache[frame_id]['sent_at']) * 1000
                        self.rtt_history.append(rtt)
                        if len(self.rtt_history) > 20:
                            self.rtt_history.pop(0)
                        
                        self.pending_frames = max(0, self.pending_frames - 1)
                        self.frames_processed += 1
                        
                        # Get detections
                        detections = data.get('detections', [])
                        
                        # Update cache via callback
                        await self._update_cache_with_detections(detections)
                        
                        # Store results for display
                        self.latest_results = {
                            'frame': self.results_cache[frame_id]['frame'],
                            'detections': detections,
                            'process_time': data.get('process_time', 0),
                            'rtt': rtt
                        }
                        
                        # Cleanup
                        if len(self.results_cache) > 30:
                            oldest = min(self.results_cache.keys())
                            del self.results_cache[oldest]
                
                elif data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                
                elif data.get('type') == 'backpressure':
                    logger.warning("Server backpressure signal received")
                    self.max_pending = max(5, self.max_pending - 1)
                    
            except json.JSONDecodeError:
                logger.error("Failed to decode message")
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logger.error(f"Error receiving results: {e}")
                break
    
    async def _display_loop(self):
        """Display frames with Supervision annotations"""
        while self.running:
            try:
                if self.latest_results and 'frame' in self.latest_results:
                    # Copy the latest frame
                    frame = self.latest_results['frame'].copy()

                    # Convert server results to Supervision format
                    detections = self._detections_from_results(
                        self.latest_results.get('detections', []),
                        frame.shape
                    )

                    # Create labels for each detection
                    labels = self._create_labels(detections)

                    # Draw bounding boxes
                    frame = self.box_annotator.annotate(
                        scene=frame,
                        detections=detections
                    )

                    # Draw labels
                    frame = self.label_annotator.annotate(
                        scene=frame,
                        detections=detections,
                        labels=labels
                    )

                    # Only draw trace paths if tracker IDs are present
                    if getattr(detections, 'tracker_id', None) is not None:
                        frame = self.trace_annotator.annotate(
                            scene=frame,
                            detections=detections
                        )

                    # Update FPS monitor
                    self.fps_monitor.tick()
                    fps_text = f"FPS: {self.fps_monitor.fps:.1f}"

                    # Build performance stats overlay
                    stats = [
                        fps_text,
                        f"RTT: {self.rtt_history[-1]:.1f}ms" if self.rtt_history else "RTT: N/A",
                        f"Quality: {self.current_quality}",
                        f"Pending: {self.pending_frames}/{self.max_pending}",
                        f"Tracked: {len(detections)}"
                    ]
                    y_offset = 30
                    for stat in stats:
                        cv2.putText(
                            frame, stat, (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
                        )
                        y_offset += 25

                    # Show the annotated frame
                    cv2.imshow('Felix Detector - Supervision', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        logger.info("User pressed 'q' to quit display")
                        break

                await asyncio.sleep(0.016)  # ~60 FPS display

            except Exception as e:
                logger.error(f"Display error: {e}")
                await asyncio.sleep(1)
    
    async def _print_stats(self):
        """Print performance statistics"""
        while self.running:
            await asyncio.sleep(5)
            
            current_time = time.time()
            elapsed = current_time - self.last_fps_time
            
            if elapsed > 0:
                send_fps = self.frames_sent / elapsed
                process_fps = self.frames_processed / elapsed
                
                avg_rtt = sum(self.rtt_history) / len(self.rtt_history) if self.rtt_history else 0
                
                logger.info(f"Stats - Send FPS: {send_fps:.1f}, Process FPS: {process_fps:.1f}, "
                           f"Pending: {self.pending_frames}/{self.max_pending}, "
                           f"RTT: {avg_rtt:.1f}ms, Quality: {self.current_quality}")
                
                # Auto-adjust flow control
                if process_fps > send_fps * 0.95 and self.pending_frames < self.max_pending * 0.5:
                    self.max_pending = min(20, self.max_pending + 1)
                elif process_fps < send_fps * 0.8:
                    self.max_pending = max(5, self.max_pending - 1)
                
                self.last_fps_time = current_time
                self.frames_sent = 0
                self.frames_processed = 0

# Example usage function (optional)
async def main():
    # Example of how to use the client with a cache
    from core.cache import VisualContextCache
    
    # Create cache
    cache = VisualContextCache()
    
    # Create client with cache callback
    client = FlowControlledClient(
        server_uri="ws://localhost:8080",
        target_fps=30,
        cache_callback=cache.update_from_client  # Pass the cache update method
    )
    
    try:
        await client.start()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await client.stop()

if __name__ == "__main__":
    asyncio.run(main())