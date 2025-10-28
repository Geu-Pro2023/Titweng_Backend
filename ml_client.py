import numpy as np
from typing import Optional
import tempfile
import os
from gradio_client import Client

class MLModelClient:
    def __init__(self):
        self.yolo_api = "Geuaguto/titweng-yolo-detector"
        self.siamese_api = "Geuaguto/titweng-siamese-embedder"
        self.yolo_client = None
        self.siamese_client = None
        print("✅ ML client initialized (lazy loading)")
    
    def _get_yolo_client(self):
        if self.yolo_client is None:
            try:
                self.yolo_client = Client(self.yolo_api)
                print(f"✅ YOLO client connected: {self.yolo_api}")
            except Exception as e:
                print(f"⚠️ YOLO client failed: {e}")
                self.yolo_client = False
        return self.yolo_client if self.yolo_client is not False else None
    
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
        """YOLO nose detection via Gradio Client"""
        try:
            client = self._get_yolo_client()
            if not client:
                return None
            
            # Save bytes to temp file for gradio_client
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            # Use gradio_client exactly like test
            yolo_result = client.predict(
                image=tmp_path,
                api_name="/predict"
            )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            nose_detected = bool(yolo_result.get("detected", False))
            if nose_detected:
                return {
                    'bbox': yolo_result.get("bbox", []),
                    'confidence': float(yolo_result.get("confidence", 0.0))
                }
            return None
                
        except Exception as e:
            print(f"YOLO API error: {e}")
            return None
    
    def extract_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Siamese embedding via Gradio Client"""
        try:
            client = self._get_siamese_client()
            if not client:
                return None
            
            # Save bytes to temp file for gradio_client
            with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name
            
            # Use gradio_client exactly like test
            siamese_result = client.predict(
                image=tmp_path,
                api_name="/predict"
            )
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            embedding_list = siamese_result.get("embedding", [])
            if embedding_list:
                embedding = np.array(embedding_list, dtype=np.float32)
                return embedding
            return None
                
        except Exception as e:
            print(f"Siamese API error: {e}")
            return None

# Global client
ml_client = MLModelClient()