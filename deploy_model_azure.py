#!/usr/bin/env python3
"""
Deploy your trained Siamese model directly to Azure
"""

# Step 1: Convert your PyTorch model to ONNX format (more portable)
# Step 2: Create Azure ML endpoint or use Azure Container Instance
# Step 3: Replace ml_client with direct model inference

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T

class LocalSiameseModel:
    def __init__(self, model_path="siamese_model_final.pth"):
        """Load your trained model directly"""
        self.device = torch.device("cpu")  # Use CPU for Azure deployment
        self.model = self._load_model(model_path)
        self.transform = T.Compose([
            T.Resize((128, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        print("✅ Local Siamese model loaded successfully")
    
    def _load_model(self, model_path):
        """Load your trained Siamese network"""
        # Import your model architecture here
        from your_model_file import SiameseNetwork  # Replace with actual import
        
        model = SiameseNetwork(embedding_dim=256)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def extract_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Extract embedding using your trained model"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Apply same transforms as training
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract embedding
            with torch.no_grad():
                embedding = self.model.forward_one(image_tensor)
                embedding = embedding.cpu().numpy().flatten()
            
            print(f"✅ Model embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
            return embedding
            
        except Exception as e:
            print(f"❌ Model inference failed: {e}")
            return None

# Usage: Replace ml_client import in main.py
# ml_client = LocalSiameseModel("path/to/your/model.pth")