import numpy as np
import requests
import os
from typing import Optional
import tempfile
import base64

class MLModelClient:
    def __init__(self):
        self.hf_token = os.getenv("HUGGINGFACE_API_TOKEN")
        self.model_url = "https://api-inference.huggingface.co/models/Geuaguto/titweng-siamese-embedder"
        print("✅ ML client initialized with HF Inference API")

    def detect_nose(self, image_bytes: bytes) -> Optional[dict]:
        """Skip YOLO - images are already cropped noses"""
        return {
            'detected': True,
            'bbox': [0, 0, 100, 100],
            'confidence': 1.0
        }
    
    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Use HF Inference API instead of Gradio Client"""
        try:
            print("🧠 Using HF Inference API for embedding extraction...")
            
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_bytes).decode()
            
            payload = {
                "inputs": image_b64,
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(
                self.model_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract embedding from response
                if isinstance(result, list) and len(result) == 256:
                    embedding = np.array(result, dtype=np.float32)
                elif isinstance(result, dict) and "embedding" in result:
                    embedding = np.array(result["embedding"], dtype=np.float32)
                else:
                    print(f"❌ Unexpected response format: {type(result)}")
                    return None
                
                # Normalize embedding
                embedding = embedding / np.linalg.norm(embedding)
                
                print(f"✅ HF API embedding: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
                return embedding
                
            else:
                print(f"❌ HF API error: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ HF Inference API failed: {e}")
            return None

# Global client
ml_client = MLModelClient()