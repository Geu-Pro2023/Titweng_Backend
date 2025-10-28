import numpy as np
from typing import Optional
import tempfile
import os
from gradio_client import Client, handle_file

class MLModelClient:
    def __init__(self):
        self.siamese_api = "Geuaguto/titweng-siamese-embedder"
        self.siamese_client = None
        print("✅ ML client initialized (lazy loading)")

    def _get_siamese_client(self):
        # Skip client creation since HF Space is broken
        print(f"Skipping Siamese API (using mock): {self.siamese_api}")
        return None
    
    def detect_nose(self, image_bytes: bytes) -> Optional[dict]:
        """Skip YOLO - images are already cropped noses"""
        return {
            'detected': True,
            'bbox': [0, 0, 100, 100],  # Dummy bbox since image is already cropped
            'confidence': 1.0
        }
    
    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Generate mock embedding for testing - replace with working HF Space"""
        try:
            # TEMPORARY: Generate a mock 256-dimension normalized embedding
            # This allows the system to work while you fix your HF Space
            print("🧠 Generating mock embedding (HF Space is down)")
            
            # Create a deterministic embedding based on image hash
            import hashlib
            image_hash = hashlib.md5(image_bytes).hexdigest()
            
            # Convert hash to numbers and create 256-dim vector
            np.random.seed(int(image_hash[:8], 16))  # Use first 8 chars as seed
            mock_embedding = np.random.randn(256).astype(np.float32)
            
            # Normalize to unit vector (like your real embeddings)
            mock_embedding = mock_embedding / np.linalg.norm(mock_embedding)
            
            print(f"✅ Mock embedding generated: shape={mock_embedding.shape}, norm={np.linalg.norm(mock_embedding):.3f}")
            return mock_embedding
                
        except Exception as e:
            print(f"Mock embedding error: {e}")
            return None

# Global client
ml_client = MLModelClient()