import asyncio
import websockets
import torch
import cv2
import numpy as np
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
import queue
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('gpu_server_v2')

class OptimizedGPUServer:
    def __init__(self, felix_model_path: str, yolo_model_path: Optional[str] = None):
        logger.info("Initializing GPU Server...")
        
        # Initialize models
        from person_detector import PersonDetector
        from felix_recognizer import FelixRecognizer
        
        self.detector = PersonDetector(yolo_model_path)
        self.recognizer = FelixRecognizer(felix_model_path)
        
        # Thread pool for CPU-bound tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Processing queue with limited size
        self.process_queue = asyncio.Queue(maxsize=10)
        
        # Client tracking
        self.clients: Dict[str, Dict[str, Any]] = {}
        
        # Performance metrics
        self.total_processed = 0
        self.processing_times = []
        
        logger.info("GPU Server initialized successfully")
    
    async def handle_client(self, websocket):
        """Handle individual client connection"""
        client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_id}")
        
        # Initialize client state
        self.clients[client_id] = {
            'websocket': websocket,
            'last_seen': time.time(),
            'frames_received': 0,
            'frames_processed': 0
        }
        
        try:
            # Start processing task for this client
            process_task = asyncio.create_task(self._process_client_frames(client_id))
            
            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    
                    if data.get('type') == 'frame':
                        # Add to processing queue if not full
                        if not self.process_queue.full():
                            await self.process_queue.put({
                                'client_id': client_id,
                                'data': data,
                                'received_at': time.time()
                            })
                            self.clients[client_id]['frames_received'] += 1
                        else:
                            # Send back pressure signal
                            await websocket.send(json.dumps({
                                'type': 'backpressure',
                                'message': 'Server busy, please slow down'
                            }))
                    
                    elif data.get('type') == 'pong':
                        self.clients[client_id]['last_seen'] = time.time()
                        
                except json.JSONDecodeError:
                    logger.error(f"Invalid JSON from {client_id}")
                except Exception as e:
                    logger.error(f"Error handling message from {client_id}: {e}")
            
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client {client_id} disconnected")
        finally:
            # Cleanup
            process_task.cancel()
            if client_id in self.clients:
                del self.clients[client_id]
    
    async def _process_client_frames(self, client_id: str):
        """Process frames for a specific client"""
        while client_id in self.clients:
            try:
                # Get frame from queue (with timeout to check if client still connected)
                try:
                    item = await asyncio.wait_for(self.process_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue
                
                # Skip if not for this client
                if item['client_id'] != client_id:
                    # Put it back for the right client
                    await self.process_queue.put(item)
                    await asyncio.sleep(0.01)
                    continue
                
                # Process the frame
                start_time = time.time()
                result = await self._process_single_frame(item['data'])
                process_time = time.time() - start_time
                
                # Update metrics
                self.total_processed += 1
                self.processing_times.append(process_time)
                if len(self.processing_times) > 100:
                    self.processing_times.pop(0)
                
                self.clients[client_id]['frames_processed'] += 1
                
                # Send result back
                response = {
                    'type': 'result',
                    'frame_id': item['data'].get('id'),
                    'detections': result,
                    'process_time': process_time,
                    'queue_time': start_time - item['received_at']
                }
                
                client_ws = self.clients[client_id]['websocket']
                await client_ws.send(json.dumps(response))
                
            except Exception as e:
                logger.error(f"Error processing frame for {client_id}: {e}")
    
    async def _process_single_frame(self, frame_data: Dict[str, Any]) -> list:
        """Process a single frame and return detections"""
        try:
            # Decode frame from hex
            frame_bytes = bytes.fromhex(frame_data['data'])
            nparr = np.frombuffer(frame_bytes, np.uint8)
            
            # Decode image in thread pool
            frame = await asyncio.get_event_loop().run_in_executor(
                self.executor, cv2.imdecode, nparr, cv2.IMREAD_COLOR
            )
            
            if frame is None:
                return []
            
            # Run detection in thread pool
            person_boxes = await asyncio.get_event_loop().run_in_executor(
                self.executor, self.detector.detect, frame
            )
            
            if not person_boxes:
                return []
            
            # Run recognition for each person (limited to first 5 to prevent overload)
            results = []
            recognition_tasks = []
            
            for box in person_boxes[:5]:  # Limit to 5 people max
                task = asyncio.get_event_loop().run_in_executor(
                    self.executor, self.recognizer.is_felix, frame, box[:4]
                )
                recognition_tasks.append((task, box))
            
            # Wait for all recognitions
            for task, box in recognition_tasks:
                try:
                    is_felix, confidence = await task
                    results.append({
                        'box': [int(x) for x in box[:4]],
                        'is_felix': bool(is_felix),
                        'confidence': float(confidence),
                        'detection_confidence': float(box[4]) if len(box) > 4 else 1.0
                    })
                except Exception as e:
                    logger.error(f"Recognition error: {e}")
            
            return results
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            return []
    
    async def _monitor_clients(self):
        """Monitor and ping clients periodically"""
        while True:
            await asyncio.sleep(10)
            
            current_time = time.time()
            for client_id, client_info in list(self.clients.items()):
                try:
                    # Send custom ping
                    await client_info['websocket'].send(json.dumps({'type': 'ping'}))
                    
                    # Check if client is responsive
                    if current_time - client_info['last_seen'] > 30:
                        logger.warning(f"Client {client_id} unresponsive, closing connection")
                        await client_info['websocket'].close()
                except:
                    pass
    
    async def _print_stats(self):
        """Print server statistics"""
        while True:
            await asyncio.sleep(30)
            
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                logger.info(f"Server Stats - Clients: {len(self.clients)}, "
                           f"Total Processed: {self.total_processed}, "
                           f"Avg Process Time: {avg_time:.3f}s, "
                           f"Queue Size: {self.process_queue.qsize()}")
            
            # Print client stats
            for client_id, info in self.clients.items():
                logger.info(f"  Client {client_id}: Received: {info['frames_received']}, "
                           f"Processed: {info['frames_processed']}")

async def main():
    # Configuration
    felix_model_path = "/root/models/felix_classifier.pth"
    yolo_model_path = None  # Use default
    host = "0.0.0.0"
    port = 8080
    
    # Initialize server
    server = OptimizedGPUServer(felix_model_path, yolo_model_path)
    
    # Start background tasks
    monitor_task = asyncio.create_task(server._monitor_clients())
    stats_task = asyncio.create_task(server._print_stats())
    
    # Start WebSocket server
    logger.info(f"Starting server on ws://{host}:{port}")
    async with websockets.serve(
        server.handle_client,
        host,
        port,
        max_size=10 * 1024 * 1024,
        ping_interval=None,  # We handle pings manually
        ping_timeout=None
    ):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")