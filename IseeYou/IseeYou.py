import asyncio
import websockets
import cv2
import numpy as np
import json
import time
import supervision as sv
import traceback

class FelixTrackingClient:
    """Handles video capture, tracking and visualization"""
    
    def __init__(self, server_url="ws://localhost:8765"):
        print("\n=== Initializing FelixTrackingClient ===")
        self.server_url = server_url
        
        self.frame_count = 0
        self.tracking_cache = {}
        # For sharing the current frame between tasks
        self.current_frame = None
        self.frame_lock = asyncio.Lock()
        
        # Skip ByteTrack - use our own simple tracking
        print("Using simple direct tracking (no ByteTrack dependency)")
        
        # This will draw boxes, labels and tracking IDs on the frame
        self.box_annotator = sv.BoxAnnotator(
            thickness=3
        )
        
        print("FelixTrackingClient initialized successfully")
    
    async def capture_and_send_frames(self, websocket, video_source):
        """Capture frames and send them to the server while making them available for display"""
        print("\n=== Starting video capture ===")
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Cannot access camera")
            return
            
        try:
            frame_counter = 0
            while True:
                ret, frame = cap.read()
                frame_counter += 1
                
                if not ret:
                    print("Error: Cannot read from camera")
                    await asyncio.sleep(0.1)
                    continue
                
                # Update the current frame for display task
                async with self.frame_lock:
                    self.current_frame = frame.copy()
                
                # Encode and send the frame to the server
                result, encoded_frame = cv2.imencode(".jpg", frame)
                if not result:
                    print("Error: Couldn't encode frame")
                    continue
                    
                frame_bytes = encoded_frame.tobytes()
                await websocket.send(frame_bytes)
                
                if frame_counter % 100 == 0:
                    print(f"[CAPTURE] Sent {frame_counter} frames to server")
                
                # Small delay to control frame rate
                await asyncio.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            print(f"Error in capture_and_send_frames: {e}")
            traceback.print_exc()
        finally:
            cap.release()
            print("Camera released")
    
    async def receive_results(self, websocket):
        """Receive detection results from the server"""
        print("\n=== Starting to receive detection results ===")
        detection_counter = 0
        try:
            async for message in websocket:
                detection_counter += 1
                
                # Parse the JSON message
                detection_data = json.loads(message)
                
                # Log what we received
                felix_count = sum(1 for det in detection_data if det.get("is_felix", False))
                print(f"[RECEIVER] Frame #{detection_counter}: Received {len(detection_data)} detections ({felix_count} Felix)")
                
                # Update tracking with new detections
                self.update_tracking(detection_data)
                
        except Exception as e:
            print(f"[RECEIVER] ERROR: {e}")
            traceback.print_exc()
    
    def update_tracking(self, detections):
        """Simple tracking implementation without ByteTrack dependency"""
        # Increment the frame counter
        self.frame_count += 1
        
        # Check if we received any detections from the server
        if not detections or len(detections) == 0:
            if self.frame_count % 30 == 0:
                print(f"[TRACKING] Frame #{self.frame_count}: No detections received")
            return  # No detections to process
        
        # Process each detection
        for i, det in enumerate(detections):
            # Extract data from server detection
            x, y, w, h = det["box"]
            is_felix = det["is_felix"]  # Direct from server - true if Felix
            confidence = det["confidence"]
            
            # Generate a simple ID based on position and frame number
            # This is a simple approach - in a real system you'd want more sophisticated tracking
            new_id = self.frame_count * 1000 + i
            
            # If there's only one detection and it's close to an existing one, try to reuse the ID
            if len(detections) == 1 and len(self.tracking_cache) > 0:
                # Find closest match in existing cache
                best_match_id = None
                best_match_dist = float('inf')
                
                for track_id, track_info in self.tracking_cache.items():
                    old_x, old_y, old_w, old_h = track_info["box"]
                    # Calculate center point distance
                    old_cx, old_cy = old_x + old_w/2, old_y + old_h/2
                    new_cx, new_cy = x + w/2, y + h/2
                    
                    dist = ((old_cx - new_cx)**2 + (old_cy - new_cy)**2)**0.5
                    
                    # If within reasonable distance, it's probably the same object
                    if dist < 100 and dist < best_match_dist:  # 100 pixels is a reasonable threshold
                        best_match_dist = dist
                        best_match_id = track_id
                
                if best_match_id is not None:
                    new_id = best_match_id
            
            # Add to tracking cache
            self.tracking_cache[new_id] = {
                "box": [x, y, w, h],
                "is_felix": is_felix,  # Store the direct value from server
                "confidence": confidence,
                "last_seen": self.frame_count
            }
        
        # Clean up stale tracks
        stale_ids = []
        for track_id, info in self.tracking_cache.items():
            if self.frame_count - info["last_seen"] > 30:  # 30 frames = ~1 second
                stale_ids.append(track_id)
        
        for track_id in stale_ids:
            del self.tracking_cache[track_id]
        
        # Log tracking updates
        if self.frame_count % 30 == 0:
            felix_tracks = sum(1 for info in self.tracking_cache.values() if info["is_felix"])
            print(f"[TRACKING] Frame #{self.frame_count}: Tracking {len(self.tracking_cache)} objects ({felix_tracks} Felix)")
    
    def visualize_frame(self, frame):
        """Draw detection results on the frame with compatibility for older Supervision versions"""
        if frame is None:
            return None
                
        # Check if tracking_cache is empty; if so, return the original frame
        if not self.tracking_cache:
            if self.frame_count % 30 == 0:
                print(f"[VISUALIZE] Frame #{self.frame_count}: No tracks to visualize")
            return frame.copy()
        
        try:
            # Prepare lists to hold the converted data
            boxes = []
            confidences = []
            class_ids = []
            track_ids = []
            
            # Extract data from tracking_cache and convert format
            for track_id, track_data in self.tracking_cache.items():
                # Get box in [x, y, w, h] format
                x, y, w, h = track_data["box"]
                
                # Convert to [x1, y1, x2, y2] format for Supervision
                x1, y1, x2, y2 = x, y, x + w, y + h
                boxes.append([x1, y1, x2, y2])
                
                # Add other data
                confidences.append(track_data["confidence"])
                # FIXED: The correct mapping - is_felix true → class_id 0
                class_id = 0 if track_data["is_felix"] else 1  # 0 for Felix, 1 for Not Felix
                class_ids.append(class_id)
                track_ids.append(track_id)
            
            # Log visualization details occasionally
            if self.frame_count % 30 == 0:
                felix_boxes = sum(1 for cls_id in class_ids if cls_id == 0)  # FIXED: class_id 0 = Felix
                print(f"[VISUALIZE] Frame #{self.frame_count}: Drawing {len(boxes)} boxes ({felix_boxes} Felix)")
            
            # Create Supervision Detections object
            detections = sv.Detections(
                xyxy=np.array(boxes),
                confidence=np.array(confidences),
                class_id=np.array(class_ids),
                tracker_id=np.array(track_ids)
            )
            
            # Method 1: Draw on the frame directly without the 'labels' parameter
            try:
                frame_copy = frame.copy()
                
                # Create custom text for each detection
                for i, (det_box, track_id, conf, class_id) in enumerate(zip(boxes, track_ids, confidences, class_ids)):
                    x1, y1, x2, y2 = det_box
                    
                    # FIXED: Get label text - class_id 0 means Felix
                    person_type = "Felix" if class_id == 0 else "Not Felix"
                    label_text = f"{person_type} #{track_id}: {conf:.2f}"
                    
                    # FIXED: Draw box with correct color mapping
                    color = (0, 255, 0) if class_id == 0 else (0, 0, 255)  # Green for Felix, Red for others
                    cv2.rectangle(frame_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    
                    # Draw label text
                    cv2.putText(
                        frame_copy, 
                        label_text,
                        (int(x1), int(y1 - 10) if y1 > 20 else int(y1 + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, 
                        color, 
                        2
                    )
                
                return frame_copy
                
            except Exception as e:
                print(f"[VISUALIZE] Error in custom visualization: {e}")
                traceback.print_exc()
                
                # Just return the original frame if everything fails
                return frame.copy()
                
        except Exception as e:
            print(f"[VISUALIZE] General error: {e}")
            traceback.print_exc()
            return frame.copy()
        
    async def display_loop(self):
        """Display processed frames using the shared current_frame"""
        print("\n=== Starting display loop ===")
        window_name = 'Felix Tracking'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Get the current frame safely
                async with self.frame_lock:
                    if self.current_frame is None:
                        await asyncio.sleep(0.03)
                        continue
                    
                    frame_to_process = self.current_frame.copy()
                
                # Process and display the frame
                processed_frame = self.visualize_frame(frame_to_process)
                
                if processed_frame is not None:
                    cv2.imshow(window_name, processed_frame)
                
                # Check for key press with a short timeout
                key = cv2.waitKey(1)
                if key == ord('q'):
                    print("User pressed 'q'. Exiting...")
                    break
                
                # Small delay to not hog CPU
                await asyncio.sleep(0.03)  # ~30 FPS
                
        except Exception as e:
            print(f"Error in display_loop: {e}")
            traceback.print_exc()
        finally:
            cv2.destroyAllWindows()
            print("Display resources released")
    
    async def run(self, video_source=0):
        """Run the client"""
        # Test camera access first
        print("\n=== Testing camera access ===")
        cap_test = cv2.VideoCapture(video_source)
        if not cap_test.isOpened():
            print("Error: Cannot access camera")
            return
            
        ret, frame = cap_test.read()
        cap_test.release()
        
        if not ret:
            print("Error: Cannot read from camera")
            return
            
        print("Camera test successful")
        
        # Now try the full system
        try:
            print("\n=== Connecting to server ===")
            async with websockets.connect(
                self.server_url,
                ping_interval=20,
                ping_timeout=20
            ) as websocket:
                print("Connected to server!")
                
                # Create tasks
                capture_send_task = asyncio.create_task(
                    self.capture_and_send_frames(websocket, video_source)
                )
                receive_task = asyncio.create_task(
                    self.receive_results(websocket)
                )
                display_task = asyncio.create_task(
                    self.display_loop()
                )
                
                # Add debugging callbacks
                capture_send_task.add_done_callback(
                    lambda t: print("Capture and send task finished")
                )
                receive_task.add_done_callback(
                    lambda t: print("Receive task finished")
                )
                display_task.add_done_callback(
                    lambda t: print("Display task finished")
                )
                
                # Wait for all tasks
                try:
                    await asyncio.gather(
                        capture_send_task, 
                        receive_task, 
                        display_task
                    )
                except asyncio.CancelledError:
                    print("Tasks were cancelled")
                    
        except ConnectionRefusedError:
            print("Error: Could not connect to server. Is it running?")
        except Exception as e:
            print(f"Connection error: {e}")
            traceback.print_exc()

# Main entry point
async def main():
    print("Starting Felix Tracking Client...")
    client = FelixTrackingClient(server_url="ws://localhost:8765")
    await client.run(video_source=0)  # Use webcam

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()