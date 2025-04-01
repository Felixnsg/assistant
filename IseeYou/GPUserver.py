import asyncio
import websockets
import torch
import cv2
import numpy as np
import json
from PIL import Image
import io
from person_detector import PersonDetector
from felix_recognizer import FelixRecognizer
import traceback
import time

import asyncio
import websockets
import torch
import cv2
import numpy as np
import json
from PIL import Image
import io
import traceback
import time
import sys
import os

# Check if running in interactive mode
is_interactive = hasattr(sys, 'ps1')

print("=== Felix Detection Server ===")
print(f"Python version: {sys.version}")
print(f"Current directory: {os.getcwd()}")
print(f"Running in interactive mode: {is_interactive}")

try:
    # Try to import the detector and recognizer modules
    print("Importing modules...")
    from person_detector import PersonDetector
    from felix_recognizer import FelixRecognizer
    print("Modules imported successfully")
except Exception as e:
    print(f"ERROR importing modules: {e}")
    traceback.print_exc()
    sys.exit(1)

class FelixDetectionServer:
    """Handles GPU-accelerated detection and recognition"""
    
    def __init__(self, felix_model, yolo_model=None):
        print("\nInitializing FelixDetectionServer...")
        
        try:
            # Initialize YOLOv8 person detector
            print("Creating person detector...")
            self.detector = PersonDetector(model_path=yolo_model)
            print("Person detector created")
            
            # Initialize Felix recognizer
            print("Creating Felix recognizer...")
            self.recognizer = FelixRecognizer(felix_model)
            print("Felix recognizer created")
            
            # Stats tracking
            self.frame_count = 0
            self.total_detections = 0
            self.felix_detections = 0
            
            # Print GPU info for debugging
            if torch.cuda.is_available():
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
                print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                print("CUDA not available. Using CPU.")
                
            print("FelixDetectionServer initialized successfully")
        except Exception as e:
            print(f"ERROR initializing server: {e}")
            traceback.print_exc()
            raise
    
    async def process_frame(self, frame):
        """Process a frame and return detection results"""
        self.frame_count += 1
        print(f"\n----- Processing Frame #{self.frame_count} -----")
        
        try:
            # Run person detection with YOLOv8
            print("Running person detection...")
            person_boxes = self.detector.detect(frame)
            print(f"Detection completed - Found {len(person_boxes)} people")
            
            results = []
            for i, box in enumerate(person_boxes):
                try:
                    # Get box dimensions
                    x, y, w, h = box[:4]
                    person_box = [int(x), int(y), int(w), int(h)]
                    
                    # Run recognition
                    print(f"Running recognition for person #{i+1}...")
                    is_felix, confidence = self.recognizer.is_felix(frame, person_box)
                    
                    # Log result
                    result_type = "FELIX" if is_felix else "NOT FELIX"
                    print(f"Person #{i+1} is {result_type} (confidence: {confidence:.3f})")
                    
                    if is_felix:
                        self.felix_detections += 1
                    
                    # Add to results
                    results.append({
                        "box": person_box,
                        "is_felix": bool(is_felix),
                        "confidence": float(confidence)
                    })
                except Exception as e:
                    print(f"Error processing detection #{i+1}: {e}")
                    traceback.print_exc()
            
            # Update counter
            self.total_detections += len(person_boxes)
            
            return results
        except Exception as e:
            print(f"Error in process_frame: {e}")
            traceback.print_exc()
            return []
    
    async def handle_client(self, websocket):
        """Handle a client connection"""
        try:
            print("\n+++ Client connected! +++\n")
            async for message in websocket:
                try:
                    # Decode the frame from binary data
                    frame_bytes = np.frombuffer(message, dtype=np.uint8)
                    frame = cv2.imdecode(frame_bytes, cv2.IMREAD_COLOR)
                    
                    if frame is None:
                        print("Error: Could not decode frame")
                        await websocket.send(json.dumps([]))
                        continue
                    
                    # Process the frame
                    results = await self.process_frame(frame)
                    
                    # Send back the results
                    await websocket.send(json.dumps(results))
                    print(f"Sent results for frame #{self.frame_count} with {len(results)} detections")
                    
                except Exception as e:
                    print(f"Error processing message: {e}")
                    traceback.print_exc()
                    await websocket.send(json.dumps([]))
        except Exception as e:
            print(f"Error handling client: {e}")
            traceback.print_exc()
        finally:
            print("\n--- Client disconnected ---\n")

async def main():
    print("\nStarting main function...")
    
    # Check if model file exists
    model_path = "/root/models/felix_classifier.pth"
    if not os.path.exists(model_path):
        print(f"ERROR: Model file not found at {model_path}")
        alternative_paths = [
            os.path.join(os.getcwd(), "models/felix_classifier.pth"),
            os.path.join(os.getcwd(), "felix_classifier.pth")
        ]
        print(f"Checking alternative paths: {alternative_paths}")
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found model at alternative path: {alt_path}")
                model_path = alt_path
                break
        else:
            print("ERROR: Could not find model file. Please check the path.")
            return
    
    try:
        # Create the server
        print("Creating server...")
        server = FelixDetectionServer(
            felix_model=model_path,
            yolo_model=None  # Use pretrained YOLOv8x model
        )
        print("Server created successfully")
        
        # Start the websocket server
        print("Starting websocket server...")
        async with websockets.serve(
            server.handle_client, 
            "0.0.0.0", 
            8080,
            ping_interval=20,
            ping_timeout=20,
            max_size=10*1024*1024  # 10MB message size limit
        ):
            print("\n=== Server running on ws://0.0.0.0:8080 ===")
            print("=== Waiting for client connections ===\n")
            await asyncio.Future()  # Run forever
            
    except Exception as e:
        print(f"ERROR in main: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    try:
        print("Starting asyncio.run(main())...")
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nServer stopped by user (KeyboardInterrupt)")
    except Exception as e:
        print(f"ERROR: Unhandled exception: {e}")
        traceback.print_exc()