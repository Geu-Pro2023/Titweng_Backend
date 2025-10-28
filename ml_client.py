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
        if self.siamese_client is None:
            try:
                self.siamese_client = Client(self.siamese_api)
                print(f"✅ Siamese client connected: {self.siamese_api}")
            except Exception as e:
                print(f"⚠️ Siamese client failed: {e}")
                self.siamese_client = False
        return self.siamese_client if self.siamese_client is not False else None
    
    def detect_nose(self, image_bytes: bytes) -> Optional[dict]:
        """Skip YOLO - images are already cropped noses"""
        return {
            'detected': True,
            'bbox': [0, 0, 100, 100],  # Dummy bbox since image is already cropped
            'confidence': 1.0
        }
    
    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Siamese embedding - EXACT copy of test_model_outputs.py"""
        try:
            client = self._get_siamese_client()
            if not client:
                return None
            
            # Save bytes to temp file for handle_file()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            # EXACT same as test: use handle_file() and /predict
            siamese_result = client.predict(
                image=handle_file(tmp_path),
                api_name="/predict"
            )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # EXACT same parsing as test
            embedding_list = siamese_result.get("embedding", [])
            if embedding_list:
                embedding = np.array(embedding_list, dtype=np.float32)
                # Verify it's 256 dimensions and normalized like test
                if embedding.shape[0] == 256:
                    return embedding
            return None
                
        except Exception as e:
            print(f"Siamese API error: {e}")
            return None

# Global client
ml_client = MLModelClient()