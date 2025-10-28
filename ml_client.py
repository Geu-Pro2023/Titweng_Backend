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
            
            print(f"🧠 Running Siamese Embedding Extractor... (image: {len(image_bytes)} bytes)")
            print(f"📝 Temp file: {tmp_path}")
            
            siamese_result = client.predict(
                image=handle_file(tmp_path),
                api_name="/predict"
            )
            
            print(f"📊 Raw API result: {siamese_result}")
            print(f"📊 Result type: {type(siamese_result)}")
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Check if result is None or empty
            if siamese_result is None:
                print("❌ API returned None")
                return None
            
            if not isinstance(siamese_result, dict):
                print(f"❌ API returned non-dict: {type(siamese_result)}")
                return None
            
            # EXACT same parsing as test_model_outputs.py
            embedding_list = siamese_result.get("embedding", [])
            print(f"📊 Embedding list: {len(embedding_list) if embedding_list else 0} items")
            
            if not embedding_list:
                print("❌ No embedding in result")
                return None
            
            embedding = np.array(embedding_list, dtype=np.float32)
            print(f"📊 Embedding shape: {embedding.shape}")
            
            if embedding.size > 0 and embedding.shape[0] == 256:
                print(f"✅ Valid embedding extracted: {embedding.shape}")
                return embedding
            else:
                print(f"❌ Invalid embedding dimensions: {embedding.shape}")
                return None
                
        except Exception as e:
            print(f"❌ Siamese API error: {e}")
            import traceback
            print(f"🔍 Full traceback: {traceback.format_exc()}")
            return None

# Global client
ml_client = MLModelClient()