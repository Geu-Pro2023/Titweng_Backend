# tests/test_model_outputs.py
import os
import numpy as np
import json
from gradio_client import Client, handle_file

# ----------------------------
# Configuration
# ----------------------------
YOLO_API = "Geuaguto/titweng-yolo-detector"
SIAMESE_API = "Geuaguto/titweng-siamese-embedder"
EXPECTED_DIM = 256

# Path to your cropped test nose image
TEST_IMAGE_PATH = os.path.join(os.path.dirname(__file__), "test_nose.jpg")

# ----------------------------
# Load APIs
# ----------------------------
print(f"Loaded YOLO API: {YOLO_API} ✔")
yolo_client = Client(YOLO_API)
print(f"Loaded Siamese API: {SIAMESE_API} ✔")
siamese_client = Client(SIAMESE_API)

# ----------------------------
# 1. Detect nose with YOLO
# ----------------------------
print("\n🧠 Running YOLO detector...")
yolo_result = yolo_client.predict(
    image=handle_file(TEST_IMAGE_PATH),
    api_name="/predict"
)

nose_detected = bool(yolo_result.get("detected", False))
bbox = yolo_result.get("bbox", [])
confidence = float(yolo_result.get("confidence", 0.0))

# ----------------------------
# 2. Generate embedding
# ----------------------------
print("🧠 Running Siamese Embedding Extractor...")
siamese_result = siamese_client.predict(
    image=handle_file(TEST_IMAGE_PATH),
    api_name="/predict"
)

embedding_list = siamese_result.get("embedding", [])
embedding = np.array(embedding_list, dtype=np.float32)
embedding_dim = int(embedding.shape[0]) if embedding.size > 0 else 0
embedding_norm = float(np.linalg.norm(embedding)) if embedding.size > 0 else 0.0
embedding_generated = bool(embedding.size > 0)
embedding_normalized = bool(np.isclose(embedding_norm, 1.0)) if embedding.size > 0 else False

# ----------------------------
# 3. Database compatibility
# ----------------------------
dimension_match = embedding_dim == EXPECTED_DIM
pgvector_format = "[" + ",".join(map(str, embedding.tolist())) + "]" if embedding.size > 0 else []

# ----------------------------
# 4. Output result
# ----------------------------
output = {
    "success": embedding_generated,
    "yolo_detector": {
        "nose_detected": nose_detected,
        "bbox": bbox,
        "confidence": confidence
    },
    "siamese_embedder": {
        "embedding_generated": embedding_generated,
        "embedding_dimension": embedding_dim,
        "embedding_type": str(type(embedding)),
        "embedding_sample": embedding[:5].tolist() if embedding.size > 0 else [],
        "embedding_norm": round(embedding_norm, 6),
        "embedding_normalized": embedding_normalized
    },
    "database_compatibility": {
        "expected_dimension": EXPECTED_DIM,
        "dimension_match": dimension_match,
        "pgvector_format": pgvector_format
    }
}

print("\n✅ Test Result:")
print(json.dumps(output, indent=2))