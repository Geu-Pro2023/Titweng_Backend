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
        """Use Gradio Client like your working local test"""
        try:
            print("🧠 Running Siamese Embedding Extractor...")
            
            # Use Gradio Client (same as your working test)
            client = self._get_siamese_client()
            
            # Save bytes to temp file for handle_file()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            try:
                # EXACT same call as your working test
                siamese_result = client.predict(
                    image=handle_file(tmp_path),
                    api_name="/predict"
                )
                
                print(f"Raw result: {siamese_result}")
                print(f"Result type: {type(siamese_result)}")
                
                # Handle your Space's actual format
                embedding_list = None
                
                if isinstance(siamese_result, dict):
                    if "embedding" in siamese_result:
                        embedding_list = siamese_result["embedding"]
                    elif "data" in siamese_result:
                        embedding_list = siamese_result["data"]
                elif isinstance(siamese_result, list):
                    embedding_list = siamese_result
                
                if embedding_list and len(embedding_list) == 256:
                    embedding = np.array(embedding_list, dtype=np.float32)
                    print(f"✅ Embedding extracted: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
                    return embedding
                else:
                    print(f"Invalid embedding format: {len(embedding_list) if embedding_list else 0} dimensions")
                    return None
                    
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.unlink(tmp_path)
                
        except Exception as e:
            print(f"Siamese API error: {e}")
            return None

# Global client
ml_client = MLModelClient()