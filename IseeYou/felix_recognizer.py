import torch
import torchvision.transforms as transforms
from PIL import Image
from torch import nn
from facenet_pytorch import InceptionResnetV1
import traceback


class FelixClassifier(nn.Module):
    def __init__(self, base_model):
        super(FelixClassifier, self).__init__()
        self.model = base_model
        self.classifier = nn.Linear(512, 2)
    
    def forward(self, x):
        embedding = self.model(x)
        output = self.classifier(embedding)
        return output



class FelixRecognizer:
    """Recognizes if a detected person is Felix with improved reliability"""
    
    def __init__(self, model_path):
        print("\n=== Initializing FelixRecognizer ===")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"• Using device: {self.device}")
        
        try:
            print(f"• Loading base model...")
            # Create a fresh base model and move it to the device immediately
            base_model = InceptionResnetV1(pretrained='vggface2')
            base_model = base_model.to(self.device)
            print("• Base model loaded successfully")
            
            print(f"• Creating classifier...")
            # Create the classifier and move it to the device
            self.model = FelixClassifier(base_model)
            self.model = self.model.to(self.device)
            
            print(f"• Loading weights from {model_path}...")
            # Load the state dictionary
            try:
                # Try loading full model first
                model_data = torch.load(model_path, map_location=self.device)
                
                # Check if it's a state dict or full model
                if isinstance(model_data, FelixClassifier):
                    print("• Found complete model")
                    self.model = model_data
                elif isinstance(model_data, dict):
                    print("• Found state dictionary")
                    self.model.load_state_dict(model_data)
                else:
                    print(f"• Unknown model format: {type(model_data)}")
            except Exception as e:
                print(f"• Warning: Error loading model weights: {e}")
                print("• Using base model only")
            
            # Final move to device to ensure everything is on the right device
            self.model = self.model.to(self.device)
            
            # Set to evaluation mode
            self.model.eval()
            print("• Model in evaluation mode")
            
            # Verify model device
            param_device = next(self.model.parameters()).device
            print(f"• Model parameters on device: {param_device}")
            
            # Define the transforms
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            print("• Transform pipeline created")
            
            # Set fallback mode to False initially
            self.fallback_mode = False
            print("=== FelixRecognizer initialization complete ===\n")
            
        except Exception as e:
            print(f"!!! Error initializing FelixRecognizer: {e}")
            print("!!! Entering fallback mode - will return default values")
            self.fallback_mode = True
            traceback.print_exc()
    
    def is_felix(self, frame, box):
        """
        Check if the person in the box is Felix with improved error handling
        Returns: (is_felix, confidence)
        """
        # If in fallback mode, return default values
        if self.fallback_mode:
            return False, 0.0
            
        try:
            # Extract the person crop from the frame
            x, y, w, h = box
            person_img = frame[y:y+h, x:x+w]
            
            # Handle empty or invalid crops
            if person_img.size == 0 or person_img.shape[0] == 0 or person_img.shape[1] == 0:
                print("Warning: Empty person crop")
                return False, 0.0
            
            # Convert to PIL Image for transforms
            person_pil = Image.fromarray(person_img)
            
            # Apply transforms and add batch dimension
            tensor = self.transform(person_pil).unsqueeze(0)
            
            # Ensure tensor is on the right device
            tensor = tensor.to(self.device)
            
            # Get prediction
            with torch.no_grad():
                output = self.model(tensor)
                probabilities = torch.softmax(output, dim=1)
                felix_prob = probabilities[0][0].item()  # Assuming class 1 is Felix
            
            # Determine if it's Felix based on threshold
            is_felix = felix_prob > 0.4 # You can adjust this threshold
            
            return is_felix, felix_prob
            
        except Exception as e:
            print(f"Error in is_felix: {e}")
            # Enter fallback mode if we hit a critical error
            if "Expected all tensors to be on the same device" in str(e):
                print("Device mismatch error - entering fallback mode")
                self.fallback_mode = True
            return False, 0.0