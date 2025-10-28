#!/usr/bin/env python3
"""
Test script to verify HF Space is working properly
"""
import requests
import json
from gradio_client import Client, handle_file
import tempfile
import time

def test_hf_space():
    space_name = "Geuaguto/titweng-siamese-embedder"
    space_url = f"https://{space_name.replace('/', '-')}.hf.space"
    
    print(f"🧪 Testing HF Space: {space_name}")
    print(f"🌐 Space URL: {space_url}")
    print("=" * 60)
    
    # Test 1: Check if Space is accessible via HTTP
    print("1️⃣ Testing HTTP accessibility...")
    try:
        response = requests.get(space_url, timeout=10)
        print(f"   ✅ HTTP Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   ✅ Space is accessible")
        else:
            print(f"   ⚠️ Unexpected status code")
    except Exception as e:
        print(f"   ❌ HTTP Error: {e}")
    
    # Test 2: Check Space API info
    print("\n2️⃣ Testing Space API info...")
    try:
        info_url = f"{space_url}/info"
        response = requests.get(info_url, timeout=10)
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            try:
                info = response.json()
                print(f"   ✅ API info retrieved successfully")
                print(f"   Available endpoints: {list(info.get('named_endpoints', {}).keys())}")
            except json.JSONDecodeError:
                print(f"   ❌ Invalid JSON in API info")
        else:
            print(f"   ❌ Failed to get API info")
    except Exception as e:
        print(f"   ❌ API Info Error: {e}")
    
    # Test 3: Try Gradio Client connection
    print("\n3️⃣ Testing Gradio Client connection...")
    try:
        print("   Connecting to Gradio Client...")
        client = Client(space_name)
        print("   ✅ Gradio Client connected successfully!")
        
        # Test 4: Try a prediction with test image
        print("\n4️⃣ Testing prediction with sample image...")
        
        # Download a test image
        test_image_url = "https://raw.githubusercontent.com/gradio-app/gradio/main/test/test_files/bus.png"
        img_response = requests.get(test_image_url, timeout=10)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(img_response.content)
            tmp_path = tmp_file.name
        
        print(f"   Test image saved to: {tmp_path}")
        
        # Make prediction
        result = client.predict(
            image=handle_file(tmp_path),
            api_name="/predict"
        )
        
        print("   ✅ Prediction successful!")
        print(f"   Result type: {type(result)}")
        print(f"   Result: {result}")
        
        # Check if result has expected format
        if isinstance(result, dict) and "embedding" in result:
            embedding = result["embedding"]
            if isinstance(embedding, list) and len(embedding) == 256:
                print("   ✅ Embedding format is correct (256 dimensions)")
            else:
                print(f"   ⚠️ Unexpected embedding format: {len(embedding) if isinstance(embedding, list) else 'not a list'}")
        elif isinstance(result, list) and len(result) == 256:
            print("   ✅ Direct embedding format is correct (256 dimensions)")
        else:
            print(f"   ⚠️ Unexpected result format")
        
        # Clean up
        import os
        os.unlink(tmp_path)
        
    except Exception as e:
        print(f"   ❌ Gradio Client Error: {e}")
        import traceback
        print(f"   Full traceback: {traceback.format_exc()}")
    
    print("\n" + "=" * 60)
    print("🏁 Test completed!")

if __name__ == "__main__":
    test_hf_space()