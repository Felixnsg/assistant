import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from facenet_pytorch import InceptionResnetV1
import traceback
import cv2
import numpy as np
from mtcnn import MTCNN


class EnhancedFelixClassifier(nn.Module):
    """Enhanced classifier with dropout and regularization - MUST MATCH TRAINING"""
    def __init__(self, base_model, dropout_rate=0.5, hidden_size=256):
        super(EnhancedFelixClassifier, self).__init__()
        self.base_model = base_model
        
        # Get the output size of the base model
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 160, 160)
            if torch.cuda.is_available():
                dummy_input = dummy_input.cuda()
                self.base_model = self.base_model.cuda()
            features = self.base_model(dummy_input)
            feature_size = features.shape[1]
        
        # Enhanced classifier with more dropout and batch norm
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8),  # Slightly less dropout in second layer
            nn.Linear(128, 2)
        )
    
    def forward(self, x):
        with torch.no_grad():
            features = self.base_model(x)
        x = self.classifier(features)
        return x


class FelixRecognizer:
    """Optimized FelixRecognizer for high-performance inference"""
    
    def __init__(self, model_path, use_mtcnn=False):
        print("\n=== Initializing Optimized FelixRecognizer ===")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"• Using device: {self.device}")
        
        try:
            # Initialize MTCNN once if needed (but we'll avoid using it)
            self.use_mtcnn = use_mtcnn
            if self.use_mtcnn:
                print("• Initializing MTCNN (this will slow things down!)...")
                self.mtcnn = MTCNN(device=self.device)
            else:
                print("• Skipping MTCNN for speed (recommended)")
                self.mtcnn = None
            
            print(f"• Loading base model...")
            base_model = InceptionResnetV1(pretrained='vggface2').eval()
            
            # Freeze base model parameters
            for param in base_model.parameters():
                param.requires_grad = False
            
            base_model = base_model.to(self.device)
            print("• Base model loaded successfully")
            
            print(f"• Creating enhanced classifier...")
            self.model = EnhancedFelixClassifier(base_model, dropout_rate=0.5, hidden_size=256)
            self.model = self.model.to(self.device)
            
            print(f"• Loading weights from {model_path}...")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict):
                    if 'model_state_dict' in checkpoint:
                        print("• Found model_state_dict in checkpoint")
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                        
                        # Print training info if available
                        if 'epoch' in checkpoint:
                            print(f"• Loaded model from epoch: {checkpoint['epoch']}")
                        if 'best_val_acc' in checkpoint:
                            print(f"• Best validation accuracy: {checkpoint['best_val_acc']:.4f}")
                    else:
                        print("• Loading as direct state dictionary")
                        self.model.load_state_dict(checkpoint)
                elif isinstance(checkpoint, EnhancedFelixClassifier):
                    print("• Found complete model")
                    self.model = checkpoint
                else:
                    print(f"• Unknown checkpoint format: {type(checkpoint)}")
                    raise ValueError("Unsupported checkpoint format")
                    
            except Exception as e:
                print(f"• Error loading model weights: {e}")
                raise
            
            self.model = self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode to disable dropout
            print("• Model in evaluation mode")
            
            # Optimized transform pipeline - changed to 160x160 to match training
            self.transform = transforms.Compose([
                transforms.Resize((160, 160)),  # Changed from 224x224 to match training
                transforms.ToTensor(),
            ])
            
            # Store normalization parameters as tensors on GPU
            self.norm_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(self.device)
            self.norm_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(self.device)
            
            self.fallback_mode = False
            print("=== FelixRecognizer initialization complete ===\n")
            
        except Exception as e:
            print(f"!!! Error initializing FelixRecognizer: {e}")
            print("!!! Entering fallback mode")
            self.fallback_mode = True
            traceback.print_exc()
    
    def preprocess_fast(self, frame, box):
        """Fast preprocessing without MTCNN"""
        try:
            x, y, w, h = [int(v) for v in box]
            
            # Add small margin to include more context
            margin = int(min(w, h) * 0.1)  # 10% margin
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            # Crop the person/face region
            crop = frame[y1:y2, x1:x2]
            
            # Convert BGR to RGB (opencv uses BGR)
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            
            # Convert to PIL for torchvision transforms
            pil_img = Image.fromarray(crop_rgb)
            
            return pil_img
            
        except Exception as e:
            print(f"Error in preprocess_fast: {e}")
            return None
    
    def preprocess_with_mtcnn(self, frame, box):
        """Slower preprocessing with MTCNN face detection"""
        try:
            x, y, w, h = [int(v) for v in box]
            
            # Add margin
            margin = 20
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            
            # Crop person region
            person_img = frame[y1:y2, x1:x2]
            person_rgb = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            
            # Detect faces with pre-initialized MTCNN
            faces = self.mtcnn.detect_faces(person_rgb)
            
            if len(faces) == 0:
                # Fall back to whole person crop
                face_img = person_rgb
            else:
                # Use largest face
                largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
                fx, fy, fw, fh = largest_face['box']
                
                # Extract face with margin
                face_margin = 10
                fx1 = max(0, fx - face_margin)
                fy1 = max(0, fy - face_margin)
                fx2 = min(person_rgb.shape[1], fx + fw + face_margin)
                fy2 = min(person_rgb.shape[0], fy + fh + face_margin)
                
                face_img = person_rgb[fy1:fy2, fx1:fx2]
            
            return Image.fromarray(face_img)
            
        except Exception as e:
            print(f"Error in preprocess_with_mtcnn: {e}")
            return None
    
    @torch.no_grad()
    def is_felix(self, frame, box):
        """Optimized inference with minimal overhead"""
        if self.fallback_mode:
            return False, 0.0
        
        try:
            # Choose preprocessing method
            if self.use_mtcnn and self.mtcnn is not None:
                person_img = self.preprocess_with_mtcnn(frame, box)
            else:
                person_img = self.preprocess_fast(frame, box)
            
            if person_img is None:
                return False, 0.0
            
            # Apply transforms
            tensor = self.transform(person_img).unsqueeze(0).to(self.device)
            
            # Normalize on GPU (faster than CPU)
            tensor = (tensor - self.norm_mean) / self.norm_std
            
            # Get prediction
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            
            # Note: Class indices might be swapped - adjust based on your training
            # If Felix is class 1 in training:
            felix_prob = probabilities[0][1].item()  # Changed from [0][0] to [0][1]
            
            is_felix = felix_prob > 0.5
            
            return is_felix, felix_prob
            
        except Exception as e:
            print(f"Error in is_felix: {e}")
            return False, 0.0
    
    @torch.no_grad()
    def is_felix_batch(self, frame, boxes):
        """Process multiple people in a single batch (even faster!)"""
        if self.fallback_mode or len(boxes) == 0:
            return [(False, 0.0) for _ in boxes]
        
        try:
            # Preprocess all images
            images = []
            valid_indices = []
            
            for i, box in enumerate(boxes):
                if self.use_mtcnn and self.mtcnn is not None:
                    img = self.preprocess_with_mtcnn(frame, box)
                else:
                    img = self.preprocess_fast(frame, box)
                
                if img is not None:
                    tensor = self.transform(img)
                    images.append(tensor)
                    valid_indices.append(i)
            
            if not images:
                return [(False, 0.0) for _ in boxes]
            
            # Stack into batch and move to GPU
            batch = torch.stack(images).to(self.device)
            
            # Normalize on GPU
            batch = (batch - self.norm_mean) / self.norm_std
            
            # Get predictions for entire batch
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            
            # Prepare results
            results = [(False, 0.0) for _ in boxes]
            for idx, valid_idx in enumerate(valid_indices):
                # Note: Class indices might be swapped - adjust based on your training
                felix_prob = probabilities[idx][1].item()  # Changed from [idx][0] to [idx][1]
                is_felix = felix_prob > 0.5
                results[valid_idx] = (is_felix, felix_prob)
            
            return results
            
        except Exception as e:
            print(f"Error in is_felix_batch: {e}")
            return [(False, 0.0) for _ in boxes]