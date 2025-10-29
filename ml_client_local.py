import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
import io
from PIL import Image
import torchvision.transforms as T
from siamese_model import SiameseNetwork

class LocalMLClient:
    def __init__(self):
        self.device = torch.device("cpu")  # Use CPU for Azure deployment
        self.model = None
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        self._load_model()
        print("✅ Local ML client initialized")

    def _load_model(self):
        """Load your trained Siamese model"""
        try:
            model_path = "siamese_model_final.pth"
            
            # Initialize model architecture
            self.model = SiameseNetwork(embedding_dim=256, pretrained=False)
            
            # Load trained weights
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print("✅ Siamese model loaded successfully")
            
        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            print("⚠️ Model file 'siamese_model_final.pth' not found")
            self.model = None
    
    def detect_nose(self, image_bytes: bytes) -> Optional[dict]:
        """Skip YOLO - images are already cropped noses"""
        return {
            'detected': True,
            'bbox': [0, 0, 100, 100],
            'confidence': 1.0
        }
    
    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract embedding using your trained Siamese model"""
        if self.model is None:
            print("❌ Model not loaded, cannot extract embedding")
            return None
            
        try:
            print("🧠 Extracting embedding with trained Siamese model...")
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Apply same transforms as training
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract embedding using your trained model
            with torch.no_grad():
                embedding = self.model.forward_one(image_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            print(f"✅ Trained model embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.6f}")
            return embedding
            
        except Exception as e:
            print(f"❌ Embedding extraction failed: {e}")
            return None

# Global client
ml_client = LocalMLClient()