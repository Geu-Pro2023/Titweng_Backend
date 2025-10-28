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
            print(f"Loaded Siamese API: {self.siamese_api} ✔")
            self.siamese_client = Client(self.siamese_api)
        return self.siamese_client
    
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
            
            # Save bytes to temp file for handle_file()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            print("🧠 Running Siamese Embedding Extractor...")
            
            # Add retry mechanism for flaky HF API
            for attempt in range(3):
                try:
                    siamese_result = client.predict(
                        image=handle_file(tmp_path),
                        api_name="/predict"
                    )
                    break
                except Exception as retry_error:
                    print(f"Attempt {attempt + 1} failed: {retry_error}")
                    if attempt == 2:  # Last attempt
                        raise retry_error
                    import time
                    time.sleep(1)  # Wait 1 second before retry
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # EXACT same parsing as test_model_outputs.py
            embedding_list = siamese_result.get("embedding", [])
            embedding = np.array(embedding_list, dtype=np.float32)
            embedding_dim = int(embedding.shape[0]) if embedding.size > 0 else 0
            embedding_norm = float(np.linalg.norm(embedding)) if embedding.size > 0 else 0.0
            embedding_generated = bool(embedding.size > 0)
            embedding_normalized = bool(np.isclose(embedding_norm, 1.0)) if embedding.size > 0 else False
            
            if embedding.size > 0 and embedding.shape[0] == 256:
                return embedding
            return None
                
        except Exception as e:
            print(f"Siamese API error: {e}")
            return None

# Global client
ml_client = MLModelClient()