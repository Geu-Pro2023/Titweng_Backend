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
        """Siamese embedding with retry logic"""
        for attempt in range(3):  # 3 retry attempts
            try:
                client = self._get_siamese_client()
                if not client:
                    print(f"Attempt {attempt + 1}: Client not available")
                    continue
                
                # Save bytes to temp file for handle_file()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_path = tmp_file.name
                
                try:
                    # EXACT same as test: use handle_file() and /predict
                    siamese_result = client.predict(
                        image=handle_file(tmp_path),
                        api_name="/predict"
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp_path)
                    
                    # Check if result is valid
                    if not siamese_result or not isinstance(siamese_result, dict):
                        print(f"Attempt {attempt + 1}: Invalid result format")
                        continue
                    
                    # EXACT same parsing as test
                    embedding_list = siamese_result.get("embedding", [])
                    if embedding_list and len(embedding_list) == 256:
                        embedding = np.array(embedding_list, dtype=np.float32)
                        print(f"✅ Embedding extracted successfully on attempt {attempt + 1}")
                        return embedding
                    else:
                        print(f"Attempt {attempt + 1}: Invalid embedding dimensions: {len(embedding_list) if embedding_list else 0}")
                        
                except Exception as api_error:
                    print(f"Attempt {attempt + 1}: API call failed: {api_error}")
                    # Clean up temp file on error
                    if os.path.exists(tmp_path):
                        os.unlink(tmp_path)
                    
                    # Reset client for next attempt
                    self.siamese_client = None
                    
            except Exception as e:
                print(f"Attempt {attempt + 1}: General error: {e}")
        
        print("❌ All embedding extraction attempts failed")
        return None

# Global client
ml_client = MLModelClient()