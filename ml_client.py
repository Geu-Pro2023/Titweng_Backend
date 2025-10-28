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
        """Siamese embedding with direct HTTP API call to force JSON"""
        try:
            # Try Gradio client first
            try:
                client = self._get_siamese_client()
                
                # Save bytes to temp file for handle_file()
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                    tmp_file.write(image_bytes)
                    tmp_path = tmp_file.name
                
                print("🧠 Running Siamese Embedding Extractor...")
                siamese_result = client.predict(
                    image=handle_file(tmp_path),
                    api_name="/predict"
                )
                
                # Clean up temp file
                os.unlink(tmp_path)
                
                # EXACT same parsing as test_model_outputs.py
                embedding_list = siamese_result.get("embedding", [])
                embedding = np.array(embedding_list, dtype=np.float32)
                
                if embedding.size > 0 and embedding.shape[0] == 256:
                    return embedding
                    
            except Exception as gradio_error:
                print(f"Gradio client failed: {gradio_error}")
                
                # Fallback: Direct HTTP API call with proper headers
                print("🔄 Trying direct HTTP API call...")
                import requests
                import base64
                
                # Convert image to base64
                image_b64 = base64.b64encode(image_bytes).decode()
                
                # Direct API call with JSON headers
                response = requests.post(
                    "https://geuaguto-titweng-siamese-embedder.hf.space/api/predict",
                    json={
                        "data": [f"data:image/jpeg;base64,{image_b64}"]
                    },
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json"
                    },
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    if "data" in result and len(result["data"]) > 0:
                        embedding_data = result["data"][0]
                        if "embedding" in embedding_data:
                            embedding_list = embedding_data["embedding"]
                            embedding = np.array(embedding_list, dtype=np.float32)
                            
                            if embedding.size > 0 and embedding.shape[0] == 256:
                                print("✅ HTTP API call successful")
                                return embedding
                
                print(f"HTTP API failed: {response.status_code} - {response.text[:200]}")
            
            return None
                
        except Exception as e:
            print(f"All embedding methods failed: {e}")
            return None

# Global client
ml_client = MLModelClient()