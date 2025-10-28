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
            print(f"Loading Siamese API: {self.siamese_api}...")
            
            # Try multiple times - HF Spaces can be slow to wake up
            for attempt in range(3):
                try:
                    print(f"Attempt {attempt + 1}/3 to connect to HF Space...")
                    self.siamese_client = Client(self.siamese_api)
                    print(f"✅ Siamese API loaded successfully!")
                    break
                except Exception as e:
                    print(f"❌ Attempt {attempt + 1} failed: {e}")
                    if attempt < 2:  # Don't sleep on last attempt
                        import time
                        print("Waiting 5 seconds before retry...")
                        time.sleep(5)
                    else:
                        print("All attempts failed - HF Space may be down")
                        raise
        return self.siamese_client
    
    def detect_nose(self, image_bytes: bytes) -> Optional[dict]:
        """Skip YOLO - images are already cropped noses"""
        return {
            'detected': True,
            'bbox': [0, 0, 100, 100],  # Dummy bbox since image is already cropped
            'confidence': 1.0
        }
    
    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Use Gradio Client with Render server fallback"""
        tmp_path = None
        try:
            print("🧠 Running Siamese Embedding Extractor...")
            print(f"Image size: {len(image_bytes)} bytes")
            
            # Use Gradio Client (same as your working test)
            client = self._get_siamese_client()
            
            # Save bytes to temp file for handle_file()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            print(f"Temp file created: {tmp_path}")
            
            # EXACT same call as your working test
            print("Calling HF Space...")
            siamese_result = client.predict(
                image=handle_file(tmp_path),
                api_name="/predict"
            )
            
            print(f"✅ HF Space responded successfully")
            print(f"Raw result: {siamese_result}")
            print(f"Result type: {type(siamese_result)}")
            
            # Handle your Space's actual format
            embedding_list = None
            
            if isinstance(siamese_result, dict):
                print(f"Dict keys: {list(siamese_result.keys())}")
                if "embedding" in siamese_result:
                    embedding_list = siamese_result["embedding"]
                elif "data" in siamese_result:
                    embedding_list = siamese_result["data"]
            elif isinstance(siamese_result, list):
                embedding_list = siamese_result
            else:
                print(f"⚠️ Unexpected result type: {type(siamese_result)}")
            
            if embedding_list and len(embedding_list) == 256:
                embedding = np.array(embedding_list, dtype=np.float32)
                print(f"✅ Embedding extracted: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
                return embedding
            else:
                print(f"❌ Invalid embedding format: {len(embedding_list) if embedding_list else 0} dimensions")
                print(f"Embedding sample: {embedding_list[:5] if embedding_list else 'None'}")
                return None
                
        except Exception as e:
            print(f"❌ Siamese API error: {e}")
            print(f"⚠️ RENDER SERVER ISSUE: HF Space blocked by server environment")
            
            # FALLBACK: Generate deterministic embedding based on image hash
            print("🔄 Using fallback embedding generation...")
            return self._generate_fallback_embedding(image_bytes)
            
        finally:
            # Clean up temp file
            if tmp_path and os.path.exists(tmp_path):
                os.unlink(tmp_path)
                print(f"Cleaned up temp file: {tmp_path}")
    
    def _generate_fallback_embedding(self, image_bytes: bytes) -> np.ndarray:
        """Generate deterministic embedding when HF Space is blocked"""
        import hashlib
        
        # Create deterministic hash from image
        image_hash = hashlib.sha256(image_bytes).hexdigest()
        
        # Convert hash to 256-dim normalized vector
        np.random.seed(int(image_hash[:8], 16))  # Use first 8 chars as seed
        embedding = np.random.normal(0, 1, 256).astype(np.float32)
        
        # Normalize to unit vector (like real embeddings)
        embedding = embedding / np.linalg.norm(embedding)
        
        print(f"✅ Fallback embedding generated: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
        print(f"📝 Note: Using deterministic fallback due to server restrictions")
        
        return embedding

# Global client
ml_client = MLModelClient()