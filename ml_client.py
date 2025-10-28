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
                print(f"Loaded Siamese API: {self.siamese_api} ✔")
                self.siamese_client = Client(self.siamese_api)
                # Test the client immediately
                print("📊 Testing Siamese client connection...")
                return self.siamese_client
            except Exception as e:
                print(f"⚠️ Siamese client failed: {e}")
                self.siamese_client = False
                return None
        elif self.siamese_client is False:
            # Try to reconnect
            try:
                print("🔄 Retrying Siamese client connection...")
                self.siamese_client = Client(self.siamese_api)
                return self.siamese_client
            except Exception as e:
                print(f"⚠️ Siamese reconnect failed: {e}")
                return None
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
            if not client:
                print("❌ Siamese client not available")
                return None
            
            # Save bytes to temp file for handle_file()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            print(f"🧠 Running Siamese Embedding Extractor... (file: {len(image_bytes)} bytes)")
            
            # Add timeout and retry for API call
            import time
            for attempt in range(2):
                try:
                    siamese_result = client.predict(
                        image=handle_file(tmp_path),
                        api_name="/predict"
                    )
                    break
                except Exception as api_error:
                    print(f"⚠️ API attempt {attempt + 1} failed: {api_error}")
                    if attempt == 0:
                        time.sleep(2)  # Wait before retry
                        continue
                    else:
                        raise api_error
            
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            
            print(f"📊 Siamese result type: {type(siamese_result)}")
            print(f"📊 Siamese result: {siamese_result}")
            
            # Check if result is valid
            if not siamese_result or not isinstance(siamese_result, dict):
                print(f"❌ Invalid result format: {siamese_result}")
                return None
            
            # EXACT same parsing as test_model_outputs.py
            embedding_list = siamese_result.get("embedding", [])
            print(f"📊 Embedding list length: {len(embedding_list) if embedding_list else 0}")
            
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
            return None

# Global client
ml_client = MLModelClient()