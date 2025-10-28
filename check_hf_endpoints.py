#!/usr/bin/env python3
"""
Check what API endpoints your HF Space actually exposes
"""
import requests
import json
from gradio_client import Client

def check_hf_space_endpoints():
    space_name = "Geuaguto/titweng-siamese-embedder"
    space_url = f"https://{space_name.replace('/', '-')}.hf.space"
    
    print(f"🔍 Checking HF Space API Endpoints")
    print(f"Space: {space_name}")
    print(f"URL: {space_url}")
    print("=" * 60)
    
    # Check 1: Space config
    print("1️⃣ Checking Space config...")
    try:
        config_url = f"{space_url}/config"
        response = requests.get(config_url, timeout=10)
        if response.status_code == 200:
            config = response.json()
            print("   ✅ Config retrieved successfully")
            
            # Look for API endpoints
            if 'dependencies' in config:
                deps = config['dependencies']
                print(f"   Dependencies: {len(deps)} found")
                for i, dep in enumerate(deps):
                    if 'api_name' in dep:
                        print(f"   API {i+1}: {dep.get('api_name', 'unnamed')}")
            
            # Look for named endpoints
            if 'named_endpoints' in config:
                endpoints = config['named_endpoints']
                print(f"   Named endpoints: {list(endpoints.keys())}")
        else:
            print(f"   ❌ Config failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Config error: {e}")
    
    # Check 2: Space info
    print("\n2️⃣ Checking Space info...")
    try:
        info_url = f"{space_url}/info"
        response = requests.get(info_url, timeout=10)
        if response.status_code == 200:
            try:
                info = response.json()
                print("   ✅ Info retrieved successfully")
                
                if 'named_endpoints' in info:
                    endpoints = info['named_endpoints']
                    print(f"   Available endpoints:")
                    for name, details in endpoints.items():
                        print(f"     - {name}: {details.get('parameters', {}).keys()}")
                
            except json.JSONDecodeError:
                print("   ⚠️ Info returned non-JSON (likely HTML)")
                print(f"   Content preview: {response.text[:200]}...")
        else:
            print(f"   ❌ Info failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Info error: {e}")
    
    # Check 3: Gradio Client inspection
    print("\n3️⃣ Checking via Gradio Client...")
    try:
        client = Client(space_name)
        print("   ✅ Gradio Client connected")
        
        # Get client info
        if hasattr(client, 'endpoints'):
            print(f"   Client endpoints: {list(client.endpoints.keys())}")
        
        if hasattr(client, 'api_info'):
            print(f"   API info available: {bool(client.api_info)}")
            if client.api_info and 'named_endpoints' in client.api_info:
                endpoints = client.api_info['named_endpoints']
                print(f"   Named endpoints from client:")
                for name in endpoints.keys():
                    print(f"     - {name}")
        
    except Exception as e:
        print(f"   ❌ Gradio Client error: {e}")
    
    # Check 4: Test common API paths
    print("\n4️⃣ Testing common API paths...")
    test_paths = [
        "/api/predict",
        "/run/predict", 
        "/call/predict",
        "/predict",
        "/api/v1/predict"
    ]
    
    for path in test_paths:
        try:
            test_url = f"{space_url}{path}"
            response = requests.get(test_url, timeout=5)
            if response.status_code == 200:
                print(f"   ✅ {path}: Available (200)")
            elif response.status_code == 405:
                print(f"   ⚠️ {path}: Method not allowed (405) - endpoint exists")
            elif response.status_code == 404:
                print(f"   ❌ {path}: Not found (404)")
            else:
                print(f"   ⚠️ {path}: Status {response.status_code}")
        except Exception as e:
            print(f"   ❌ {path}: Error - {e}")
    
    print("\n" + "=" * 60)
    print("🏁 Endpoint check completed!")

if __name__ == "__main__":
    check_hf_space_endpoints()