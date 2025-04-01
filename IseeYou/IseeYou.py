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
    
    def __init__(self, server_url="ws://localhost:8080"):
        print("\n=== Initializing FelixTrackingClient ===")
        self.server_url = server_url
        
        self.frame_count = 0
        self.tracking_cache = {}
        # For sharing the current frame between tasks
        self.current_frame = None
        self.frame_lock = asyncio.Lock()
        
        # For fallback visualization when tracking doesn't work
        self.raw_detections = []
        
        # Skip ByteTrack - use our own simple tracking
        print("Using simple direct tracking (no ByteTrack dependency)")
        
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
                
                # Store raw detections for fallback visualization
                self.raw_detections = detection_data
                
                # Log what we received
                felix_count = sum(1 for det in detection_data if det.get("is_felix", False))
                print(f"[RECEIVER] Frame #{detection_counter}: Received {len(detection_data)} detections ({felix_count} Felix)")
                
                # Update tracking with new detections
                self.update_tracking(detection_data)
                
        except Exception as e:
            print(f"[RECEIVER] ERROR: {e}")
            traceback.print_exc()
    
    def update_tracking(self, detections):
        """Tracking with Supervision's ByteTrack wrapper"""
        self.frame_count += 1

        if not detections:
            if self.frame_count % 30 == 0:
                print(f"[TRACKING] Frame #{self.frame_count}: No detections received")
            return

        # Debug: Print confidence values occasionally
        if self.frame_count % 30 == 0:
            confidences = [det["confidence"] for det in detections]
            print(f"[TRACKING] Debug: Detection confidences: {confidences}")

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
            print(f"[TRACKING] Frame #{self.frame_count}: Tracking {len(tracked_detections)} objects ({felix_count} Felix)")
            
            # Debug message if tracking creates no tracks
            if len(tracked_detections) == 0 and len(detections) > 0:
                print(f"[TRACKING] Warning: ByteTrack created 0 tracks from {len(detections)} detections.")
    
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
                    print(f"[VISUALIZE] Frame #{self.frame_count}: Drawing {len(self.tracked_detections)} tracked boxes ({felix_count} Felix)")
                
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
                box_annotator = sv.BoundingBoxAnnotator(
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
                print(f"[VISUALIZE] Error with tracked visualization: {e}")
                # Will fall through to fallback methods
        
        # OPTION 2: Fallback to raw detections if tracking doesn't work
        elif hasattr(self, 'raw_detections') and self.raw_detections:
            try:
                if self.frame_count % 30 == 0:
                    felix_count = sum(1 for det in self.raw_detections if det.get("is_felix", False))
                    print(f"[VISUALIZE] Frame #{self.frame_count}: Fallback to {len(self.raw_detections)} raw detections ({felix_count} Felix)")
                
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
                print(f"[VISUALIZE] Error with fallback visualization: {e}")
                # Will fall through to default
        
        # Option 3: No detections to visualize
        else:
            if self.frame_count % 30 == 0:
                print(f"[VISUALIZE] Frame #{self.frame_count}: No tracks or detections to visualize")
        
        return frame_copy
        
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
    client = FelixTrackingClient(server_url="ws://localhost:8080")
    await client.run(video_source=0)  # Use webcam

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unhandled exception: {e}")
        traceback.print_exc()