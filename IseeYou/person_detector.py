from ultralytics import YOLO
import cv2
import torch
import numpy as np
import traceback

class PersonDetector:
    """Person detector using YOLOv8"""
    
    def __init__(self, model_path=None):
        """Initialize YOLOv8 detector
        
        Args:
            model_path: Path to YOLOv8 model, or None to use pretrained model
        """
        try:
            # Use a pretrained YOLOv8 model if no model is specified
            if model_path is None:
                print("Loading pretrained YOLOv8 model...")
                self.model = YOLO("yolov8x.pt")  # Use the largest model for best accuracy
            else:
                print(f"Loading YOLOv8 model from {model_path}...")
                self.model = YOLO(model_path)
                
            # Make sure we're using CUDA since we're on an RTX 4090
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {self.device}")
            
            # Class ID for person in COCO dataset (which YOLOv8 uses by default)
            self.person_class_id = 0
            
            # Run a test inference to make sure everything works
            dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            print("YOLOv8 model loaded successfully!")
            
        except Exception as e:
            print(f"Error initializing YOLOv8 model: {e}")
            traceback.print_exc()
            raise
    
    def detect(self, frame):
        """Detect people in a frame
        
        Args:
            frame: OpenCV image (BGR format)
            
        Returns:
            List of [x, y, w, h, confidence] bounding boxes for detected people
        """
        try:
            # Run YOLOv8 inference (batch size of 1)
            results = self.model(frame, verbose=False)
            
            # Extract person detections
            person_boxes = []
            
            for result in results:
                # Get tensor data
                boxes = result.boxes
                
                # Filter for person class (class_id 0 in COCO)
                for i, class_id in enumerate(boxes.cls.cpu().numpy().astype(int)):
                    if class_id == self.person_class_id:
                        # Get box coordinates (XYXY format)
                        box = boxes.xyxy[i].cpu().numpy().astype(int)
                        x1, y1, x2, y2 = box
                        
                        # Convert to [x, y, w, h] format
                        x, y = x1, y1
                        w, h = x2 - x1, y2 - y1
                        
                        # Add confidence as the 5th element
                        confidence = boxes.conf[i].cpu().numpy().item()
                        
                        person_boxes.append([x, y, w, h, confidence])

            CONFIDENCE_THRESHOLD = 0.7  # Only keep detections with 70% or higher confidence
    
            # Filter person boxes by confidence
            filtered_person_boxes = []
            for box in person_boxes:
                confidence = box[4]  # The confidence is stored as the 5th element
                if confidence >= CONFIDENCE_THRESHOLD:
                    filtered_person_boxes.append(box)
            
            return filtered_person_boxes
                    
                    
        except Exception as e:
            print(f"Error in YOLOv8 detection: {e}")
            traceback.print_exc()
            return []  # Return empty list on error