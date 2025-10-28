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
        """Direct HTTP API call to HF Space - bypasses Gradio Client HTML issues"""
        try:
            print("🧠 Running Siamese Embedding Extractor (Direct HTTP)...")
            import requests
            import base64
            
            # Convert image to base64
            image_b64 = base64.b64encode(image_bytes).decode()
            
            # Try multiple API endpoints
            endpoints = [
                "https://geuaguto-titweng-siamese-embedder.hf.space/api/predict",
                "https://geuaguto-titweng-siamese-embedder.hf.space/run/predict",
                "https://geuaguto-titweng-siamese-embedder.hf.space/call/predict"
            ]
            
            for endpoint in endpoints:
                try:
                    print(f"Trying endpoint: {endpoint}")
                    
                    # Direct API call with proper Gradio format
                    response = requests.post(
                        endpoint,
                        json={
                            "data": [f"data:image/jpeg;base64,{image_b64}"]
                        },
                        headers={
                            "Content-Type": "application/json",
                            "Accept": "application/json"
                        },
                        timeout=30
                    )
                    
                    print(f"Response status: {response.status_code}")
                    print(f"Response headers: {dict(response.headers)}")
                    
                    if response.status_code == 200:
                        try:
                            result = response.json()
                            print(f"JSON result: {result}")
                            
                            # Handle different response formats
                            embedding_list = None
                            
                            # Format 1: {"data": [{"embedding": [...]}]}
                            if "data" in result and len(result["data"]) > 0:
                                data_item = result["data"][0]
                                if isinstance(data_item, dict) and "embedding" in data_item:
                                    embedding_list = data_item["embedding"]
                                elif isinstance(data_item, list):
                                    embedding_list = data_item
                            
                            # Format 2: {"embedding": [...]}
                            elif "embedding" in result:
                                embedding_list = result["embedding"]
                            
                            # Format 3: Direct list
                            elif isinstance(result, list):
                                embedding_list = result
                            
                            if embedding_list and len(embedding_list) == 256:
                                embedding = np.array(embedding_list, dtype=np.float32)
                                print(f"✅ Embedding extracted: shape={embedding.shape}, norm={np.linalg.norm(embedding):.3f}")
                                return embedding
                            else:
                                print(f"Invalid embedding: {len(embedding_list) if embedding_list else 0} dimensions")
                                
                        except Exception as json_error:
                            print(f"JSON parsing failed: {json_error}")
                            print(f"Raw response: {response.text[:500]}")
                    else:
                        print(f"HTTP error: {response.status_code} - {response.text[:200]}")
                        
                except Exception as endpoint_error:
                    print(f"Endpoint {endpoint} failed: {endpoint_error}")
                    continue
            
            print("❌ All API endpoints failed")
            return None
                
        except Exception as e:
            print(f"Complete embedding extraction failed: {e}")
            return None

# Global client
ml_client = MLModelClient()