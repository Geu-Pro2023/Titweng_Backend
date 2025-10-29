#!/usr/bin/env python3
"""
Debug ML model consistency
"""
import requests
import json

def test_ml_endpoint():
    """Test the ML model endpoint directly"""
    
    # Test the ML model endpoint
    url = "https://titweng-app-a3hufygwcphxhkc2.canadacentral-01.azurewebsites.net/test-ml-models"
    
    try:
        response = requests.post(url)
        print("ML Test Response:")
        print(json.dumps(response.json(), indent=2))
        
        # Check if embeddings are consistent
        result = response.json()
        if "embedding_test" in result:
            emb = result["embedding_test"]
            if emb and len(emb) == 256:
                print(f"\n✅ Embedding extracted: {len(emb)} dimensions")
                print(f"Sample values: {emb[:5]}")
                print(f"Embedding norm: {sum(x*x for x in emb)**0.5:.4f}")
            else:
                print(f"\n❌ Invalid embedding: {emb}")
        
    except Exception as e:
        print(f"Error testing ML endpoint: {e}")

if __name__ == "__main__":
    test_ml_endpoint()