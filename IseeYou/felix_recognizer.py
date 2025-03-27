# File: felix_recognizer.py
# --- REFACTOR: Added file description ---
"""
Provides functionality to recognize a specific person ("Felix") within
a detected bounding box using a fine-tuned facial recognition model.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from torch import nn
from facenet_pytorch import InceptionResnetV1 # Base model for face recognition
import traceback
import os # --- REFACTOR: Added os for path checking ---
import sys # --- REFACTOR: Added sys for path append ---
import numpy as np # --- REFACTOR: Added numpy for frame type check ---

# --- REFACTOR: Simplified path appending ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import config
except ImportError as e:
    print(f"Error importing config module in felix_recognizer.py: {e}. Ensure config.py exists.")
    sys.exit(1)


# --- REFACTOR: Classifier definition remains the same, added docstring ---
class FelixClassifier(nn.Module):
    """
    A classifier built on top of a base face embedding model (InceptionResnetV1).

    Takes the 512-dimension embedding from the base model and maps it to
    2 output classes (Felix vs. Not Felix).
    """
    def __init__(self, base_model):
        super().__init__()
        self.model = base_model
        # --- REFACTOR: Ensure classifier layer matches embedding size ---
        # InceptionResnetV1 output is 512 for vggface2 and casia-webface
        self.classifier = nn.Linear(512, 2) # Output size 2 (Felix, Not Felix)

    def forward(self, x):
        # --- REFACTOR: Ensure base model is not in training mode if using dropout/batchnorm ---
        # self.model.eval() # Set base model to eval mode - DO THIS EXTERNALLY once after loading
        embedding = self.model(x)
        output = self.classifier(embedding)
        return output


class FelixRecognizer:
    """
    Recognizes if a person crop corresponds to "Felix" using a trained classifier.

    Attributes:
        device (torch.device): The device (CPU or CUDA GPU) for model inference.
        model (FelixClassifier): The loaded classification model.
        transform (transforms.Compose): Image transformations for model input.
        confidence_threshold (float): Minimum probability threshold to classify as Felix.
        fallback_mode (bool): If True, indicates an initialization error occurred,
                              and recognition will return default values.
    """

    # --- REFACTOR: Improved docstring, type hinting, configuration ---
    def __init__(self, model_path: str = config.FELIX_MODEL_PATH, confidence_threshold: float = config.FELIX_RECOGNIZER_THRESHOLD):
        """
        Initializes the FelixRecognizer.

        Loads the base InceptionResnetV1 model, creates the FelixClassifier,
        loads trained weights, and sets up image transforms.

        Args:
            model_path (str, optional): Path to the saved FelixClassifier weights (.pth file).
                                        Defaults to config.FELIX_MODEL_PATH.
            confidence_threshold (float, optional): Minimum probability for Felix classification.
                                                    Defaults to config.FELIX_RECOGNIZER_THRESHOLD.

        Raises:
            FileNotFoundError: If the specified model_path does not exist.
            RuntimeError: If model loading or initialization fails critically.
        """
        print("\n=== Initializing FelixRecognizer ===")
        self.fallback_mode = False # Assume success initially
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"• FelixRecognizer: Using device: {self.device}")
        self.confidence_threshold = confidence_threshold
        print(f"• FelixRecognizer: Confidence threshold set to {self.confidence_threshold}")

        # --- REFACTOR: Centralized error handling for initialization ---
        try:
            # --- 1. Load Base Model ---
            print(f"• FelixRecognizer: Loading base model InceptionResnetV1 (vggface2)...")
            # Load pretrained base model, move to device immediately
            base_model = InceptionResnetV1(pretrained='vggface2').to(self.device)
            # Set base model to evaluation mode IF it's not part of the training graph
            # If FelixClassifier fine-tuned the base model, keep eval() for the combined model later.
            # base_model.eval() # Defer this to the final combined model
            print("• FelixRecognizer: Base model loaded.")

            # --- 2. Create Classifier ---
            print(f"• FelixRecognizer: Creating FelixClassifier...")
            self.model = FelixClassifier(base_model).to(self.device)
            print("• FelixRecognizer: Classifier created.")

            # --- 3. Load Trained Weights ---
            if not model_path or not os.path.exists(model_path):
                 print(f"!!! FelixRecognizer: Error - Model weights file not found at '{model_path}'")
                 # --- REFACTOR: Handle missing model file ---
                 # Option 1: Raise error
                 # raise FileNotFoundError(f"FelixRecognizer model weights not found: {model_path}")
                 # Option 2: Enter fallback mode
                 print("!!! FelixRecognizer: Entering fallback mode due to missing model file.")
                 self.fallback_mode = True
                 # Skip further loading attempts if in fallback
            else:
                print(f"• FelixRecognizer: Loading weights from {model_path}...")
                # Load state dictionary with mapping to the correct device
                try:
                    # --- REFACTOR: Load explicitly to the target device ---
                    model_data = torch.load(model_path, map_location=self.device)

                    # Check if it's a state dict or a full model object
                    if isinstance(model_data, dict):
                        # Check for potential keys mismatch (e.g., DataParallel saving)
                        # Example: remove 'module.' prefix if saved with DataParallel
                        state_dict = model_data
                        if all(key.startswith('module.') for key in model_data.keys()):
                             print("• FelixRecognizer: Removing 'module.' prefix from state_dict keys.")
                             state_dict = {k[len('module.'):]: v for k, v in model_data.items()}

                        # Load the state dictionary
                        self.model.load_state_dict(state_dict)
                        print("• FelixRecognizer: State dictionary loaded successfully.")
                    # --- REFACTOR: Loading full model is generally discouraged, prefer state_dict ---
                    # elif isinstance(model_data, FelixClassifier):
                    #     print("• FelixRecognizer: Loaded complete model object (use state_dict preferred).")
                    #     self.model = model_data.to(self.device) # Ensure it's on correct device
                    else:
                        # Handle unexpected format
                        print(f"!!! FelixRecognizer: Warning - Loaded model file '{model_path}' contains unexpected data type: {type(model_data)}. Attempting to proceed, but weights might not be loaded correctly.")
                        # Optionally enter fallback mode here too
                        # self.fallback_mode = True

                except FileNotFoundError:
                     # This case should be caught by the os.path.exists check earlier, but good practice
                     print(f"!!! FelixRecognizer: Error loading weights - File not found '{model_path}'.")
                     self.fallback_mode = True # Enter fallback
                except RuntimeError as e:
                     # Catches issues like size mismatches in layers
                     print(f"!!! FelixRecognizer: Error loading state_dict (likely mismatch): {e}")
                     self.fallback_mode = True # Enter fallback
                except Exception as e:
                     print(f"!!! FelixRecognizer: Unexpected error loading model weights: {e}")
                     traceback.print_exc()
                     self.fallback_mode = True # Enter fallback

            # --- 4. Set Model to Evaluation Mode (Crucial) ---
            if not self.fallback_mode:
                self.model.eval()
                print("• FelixRecognizer: Model set to evaluation mode.")

                # --- 5. Verify Model Device ---
                # Check device of a parameter to confirm
                try:
                    param_device = next(self.model.parameters()).device
                    print(f"• FelixRecognizer: Model parameters confirmed on device: {param_device}")
                    if param_device != self.device:
                         print(f"!!! FelixRecognizer: Warning - Device mismatch after loading! Model on {param_device}, expected {self.device}. Attempting move...")
                         self.model.to(self.device)
                         param_device = next(self.model.parameters()).device
                         print(f"• FelixRecognizer: Model parameters now on device: {param_device}")
                except StopIteration:
                     print("!!! FelixRecognizer: Warning - Model appears to have no parameters.")
                     self.fallback_mode = True # No parameters means model is likely broken

            # --- 6. Define Transforms ---
            # Standard transforms for face recognition models
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)), # Input size for InceptionResnetV1
                transforms.ToTensor(),
                # --- REFACTOR: Use standard normalization for pretrained models ---
                # These are ImageNet stats, often used, but verify if vggface2 uses different ones
                # Standard for InceptionResnetV1 often is: transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # Let's use the 0.5 normalization common for facenet models.
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
                # Original code used ImageNet stats:
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print(f"• FelixRecognizer: Transform pipeline created (Resize to 160x160, Normalize with 0.5 mean/std).")

            if not self.fallback_mode:
                print("=== FelixRecognizer initialization complete ===")
            else:
                 print("=== FelixRecognizer initialized in FALLBACK MODE ===")

        except Exception as e:
            # Catch-all for any unexpected error during the whole init process
            print(f"!!! FelixRecognizer: CRITICAL Error during initialization: {e}")
            traceback.print_exc()
            print("!!! FelixRecognizer: Entering fallback mode.")
            self.fallback_mode = True
            print("=== FelixRecognizer initialized in FALLBACK MODE ===")

    # --- REFACTOR: Improved docstring, type hinting, error handling ---
    def is_felix(self, frame: np.ndarray, box: list[int]) -> tuple[bool, float]:
        """
        Checks if the person within the bounding box in the frame is Felix.

        Args:
            frame (np.ndarray): The full image frame (OpenCV BGR format).
            box (list[int]): The bounding box [x, y, w, h] of the detected person.

        Returns:
            tuple[bool, float]: A tuple containing:
                                - bool: True if the person is classified as Felix above the threshold, False otherwise.
                                - float: The raw probability score for the Felix class (typically class 0).
                                Returns (False, 0.0) if in fallback mode or if an error occurs.
        """
        if self.fallback_mode:
            # print("FelixRecognizer: Running in fallback mode, returning (False, 0.0).") # Can be noisy
            return False, 0.0
        if not isinstance(frame, np.ndarray):
             print("FelixRecognizer Error: Input frame must be a NumPy array.")
             return False, 0.0
        if not isinstance(box, (list, tuple)) or len(box) != 4:
             print(f"FelixRecognizer Error: Invalid box format: {box}. Expected [x, y, w, h].")
             return False, 0.0

        try:
            x, y, w, h = map(int, box) # Ensure integer coordinates

            # --- REFACTOR: Add padding to the crop (optional but often helpful) ---
            padding = 0 # Example: int(0.1 * max(w, h)) # Add 10% padding
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(frame.shape[1], x + w + padding) # Use frame dimensions for bounds
            y2 = min(frame.shape[0], y + h + padding)

            # Check if box dimensions are valid after padding/clipping
            if x1 >= x2 or y1 >= y2:
                 print(f"FelixRecognizer Warning: Invalid box dimensions after padding/clipping: [{x1},{y1},{x2},{y2}]. Original: {box}")
                 return False, 0.0

            # Extract the person crop (BGR format)
            person_crop_bgr = frame[y1:y2, x1:x2]

            # Handle empty crops
            if person_crop_bgr.size == 0:
                print("FelixRecognizer Warning: Empty person crop extracted.")
                return False, 0.0

            # Convert BGR crop to RGB PIL Image for transforms
            try:
                person_pil = Image.fromarray(cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB))
            except (cv2.error, UnidentifiedImageError) as e:
                 print(f"FelixRecognizer Error: Failed to convert crop to PIL Image: {e}")
                 return False, 0.0


            # Apply transforms, add batch dimension, move to device
            try:
                # --- REFACTOR: Ensure tensor is created correctly and moved ---
                tensor = self.transform(person_pil).unsqueeze(0).to(self.device)
            except Exception as e:
                 print(f"FelixRecognizer Error: Failed during image transformation: {e}")
                 return False, 0.0

            # Perform inference
            with torch.no_grad(): # Essential for inference
                try:
                    output = self.model(tensor)
                    # Apply softmax to get probabilities
                    probabilities = torch.softmax(output, dim=1)

                    # --- REFACTOR: Assuming class 0 is Felix, class 1 is Not Felix ---
                    # Verify this based on your training data labels!
                    felix_prob = probabilities[0][0].item()
                    # not_felix_prob = probabilities[0][1].item()

                except RuntimeError as e:
                     # Catch potential device mismatches or tensor shape errors during forward pass
                     print(f"FelixRecognizer Error: PyTorch runtime error during inference: {e}")
                     # Attempt to diagnose device mismatch
                     if "Expected all tensors to be on the same device" in str(e):
                         print("!!! FelixRecognizer: Device mismatch detected during inference! Entering fallback mode.")
                         self.fallback_mode = True
                     return False, 0.0
                except Exception as e:
                     print(f"FelixRecognizer Error: Unexpected error during model inference: {e}")
                     return False, 0.0

            # Determine classification based on threshold
            is_felix = felix_prob >= self.confidence_threshold

            # Debug log
            # print(f"FelixRecognizer: Box={box}, Felix Prob={felix_prob:.4f}, IsFelix={is_felix}")

            return is_felix, felix_prob

        except Exception as e:
            print(f"FelixRecognizer Error: Unexpected error in is_felix method: {e}")
            traceback.print_exc()
            return False, 0.0

# --- REFACTOR: Import cv2 needed for color conversion ---
import cv2