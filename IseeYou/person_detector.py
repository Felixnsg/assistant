# File: person_detector.py
# --- REFACTOR: Added file description ---
"""
Provides person detection functionality using the YOLOv8 model.

Detects persons in image frames and returns their bounding boxes and confidence scores.
"""

from ultralytics import YOLO
# import cv2 # Not used directly in this file, frame comes from caller
import torch
import numpy as np
import traceback
import os # --- REFACTOR: Added os for path checking ---
import sys # --- REFACTOR: Added sys for path append ---

# --- REFACTOR: Simplified path appending ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import config
except ImportError as e:
    print(f"Error importing config module in person_detector.py: {e}. Ensure config.py exists.")
    sys.exit(1)


class PersonDetector:
    """
    Detects persons in image frames using a YOLOv8 model.

    Attributes:
        model (YOLO): The loaded YOLOv8 model instance.
        device (torch.device): The device (CPU or CUDA GPU) used for inference.
        person_class_id (int): The class ID corresponding to 'person' in the model's dataset (usually COCO).
        confidence_threshold (float): Minimum confidence score to consider a detection valid.
    """

    # --- REFACTOR: Improved docstring, type hinting, added confidence threshold from config ---
    def __init__(self, model_path: str = config.YOLO_MODEL_PATH, confidence_threshold: float = config.PERSON_DETECTION_THRESHOLD):
        """
        Initializes the YOLOv8 person detector.

        Args:
            model_path (str, optional): Path to a specific YOLOv8 model file (.pt).
                                        If None or invalid, uses the default 'yolov8x.pt' pretrained model.
                                        Defaults to config.YOLO_MODEL_PATH.
            confidence_threshold (float, optional): Minimum confidence score for detections.
                                                    Defaults to config.PERSON_DETECTION_THRESHOLD.

        Raises:
            RuntimeError: If the YOLO model fails to load.
        """
        self.model = None
        self.confidence_threshold = confidence_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PersonDetector: Using device: {self.device}")

        resolved_model_path = model_path
        default_model = "yolov8x.pt" # Use a large default model

        # --- REFACTOR: Validate custom model path or use default ---
        if model_path and os.path.exists(model_path):
            print(f"PersonDetector: Loading custom YOLOv8 model from {model_path}...")
        else:
            if model_path:
                 print(f"PersonDetector: Warning - Custom model path '{model_path}' not found.")
            print(f"PersonDetector: Loading default pretrained YOLOv8 model '{default_model}'...")
            resolved_model_path = default_model

        try:
            self.model = YOLO(resolved_model_path)
            # --- REFACTOR: Move model to the correct device explicitly ---
            self.model.to(self.device)

            # --- REFACTOR: Determine person class ID dynamically if possible/needed ---
            # Assuming COCO dataset where 'person' is class 0.
            # If using a custom model, this might need adjustment.
            self.person_class_id = 0
            # You could potentially inspect model.names here if available:
            # if hasattr(self.model, 'names'):
            #     try:
            #         self.person_class_id = self.model.names.index('person')
            #     except (ValueError, AttributeError):
            #         print("Warning: Could not dynamically find 'person' class ID. Assuming ID 0.")
            #         self.person_class_id = 0
            # else:
            #      print("Warning: Model 'names' attribute not found. Assuming person class ID 0.")
            #      self.person_class_id = 0
            print(f"PersonDetector: Using class ID {self.person_class_id} for 'person'.")


            # --- REFACTOR: Run test inference more safely ---
            print("PersonDetector: Running initial test inference...")
            dummy_image = np.zeros((640, 480, 3), dtype=np.uint8)
            try:
                # Ensure dummy image is on the correct device if required by model input
                # dummy_image_tensor = torch.from_numpy(dummy_image).to(self.device).float() / 255.0 # Example conversion
                _ = self.model(dummy_image, verbose=False, device=self.device) # Pass device explicitly
                print("PersonDetector: YOLOv8 model loaded and test inference successful!")
            except Exception as test_e:
                 print(f"PersonDetector: Error during test inference: {test_e}")
                 # Decide if this is fatal
                 raise RuntimeError(f"YOLOv8 test inference failed: {test_e}") from test_e

        except FileNotFoundError as e:
             print(f"PersonDetector: Error - Model file '{resolved_model_path}' not found: {e}")
             traceback.print_exc()
             raise RuntimeError(f"YOLOv8 model file not found: {resolved_model_path}") from e
        except Exception as e:
            print(f"PersonDetector: Error initializing YOLOv8 model '{resolved_model_path}': {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to initialize YOLOv8 model: {e}") from e

    # --- REFACTOR: Improved docstring, type hinting, error handling ---
    def detect(self, frame: np.ndarray) -> list[list]:
        """
        Detects people in a given image frame.

        Args:
            frame (np.ndarray): The input image frame (OpenCV BGR format).

        Returns:
            list[list]: A list of detected person bounding boxes. Each inner list
                        contains [x, y, w, h, confidence], where (x, y) is the
                        top-left corner, and w, h are width and height. Returns
                        an empty list if no persons are detected above the threshold
                        or if an error occurs.
        """
        if self.model is None:
            print("PersonDetector: Error - Model not loaded, cannot detect.")
            return []
        if not isinstance(frame, np.ndarray):
             print("PersonDetector: Error - Input frame must be a NumPy array.")
             return []

        person_boxes = []
        try:
            # --- REFACTOR: Explicitly set device and disable verbose logging ---
            results = self.model(frame, verbose=False, device=self.device, conf=self.confidence_threshold, classes=[self.person_class_id])

            # --- REFACTOR: Process results more directly using ultralytics structure ---
            for result in results:
                boxes = result.boxes  # ultralytics Boxes object
                for i in range(len(boxes)):
                    class_id = int(boxes.cls[i].item()) # Get class ID
                    # Double check, though we filtered by class in the call
                    if class_id == self.person_class_id:
                        confidence = boxes.conf[i].item()
                        # Get box coordinates (XYXY format)
                        x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                        # Convert to [x, y, w, h] format
                        x, y = x1, y1
                        w, h = x2 - x1, y2 - y1

                        # Append [x, y, w, h, confidence]
                        person_boxes.append([x, y, w, h, confidence])

            # --- REFACTOR: Confidence filtering is now done in model call `conf=...` ---
            # filtered_person_boxes = [box for box in person_boxes if box[4] >= self.confidence_threshold]
            # return filtered_person_boxes
            return person_boxes # Already filtered by the model call

        except Exception as e:
            print(f"PersonDetector: Error during YOLOv8 detection: {e}")
            traceback.print_exc()
            return [] # Return empty list on error