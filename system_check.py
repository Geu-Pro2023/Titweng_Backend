#!/usr/bin/env python3
import os
import sys
import json
import requests
from sqlalchemy import create_engine, text
from gradio_client import Client

def check_database():
    """Test PostgreSQL database connection"""
    try:
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://titweng_user:titweng!.com@localhost:5432/titweng")
        if DATABASE_URL.startswith("postgres://"):
            DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            conn.execute(text("SELECT version()"))
        
        print("✅ Database: Connected")
        return True
    except Exception as e:
        print(f"❌ Database: Failed - {e}")
        return False

def check_ml_services():
    """Test Hugging Face ML services"""
    try:
        siamese_api = "Geuaguto/titweng-siamese-embedder"
        print(f"Testing Siamese API: {siamese_api}")
        client = Client(siamese_api)
        print("✅ ML Services: Siamese API connected")
        return True
    except Exception as e:
        print(f"❌ ML Services: Failed - {e}")
        return False

def check_api_endpoints():
    """Test API endpoints if running locally"""
    try:
        base_url = "http://localhost:8000"
        
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            health_data = response.json()
            print("✅ API Health:", health_data.get("status", "unknown"))
            return True
        else:
            print(f"❌ API Health: Status {response.status_code}")
            return False
    except Exception as e:
        print(f"⚠️ API Health: Not running locally - {e}")
        return False

def check_environment():
    """Check environment variables"""
    required_vars = ["DATABASE_URL"]
    optional_vars = ["ADMIN_USERNAME", "ADMIN_PASSWORD", "BREVO_API_KEY"]
    
    print("\n🔧 Environment Variables:")
    all_good = True
    
    for var in required_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"❌ {var}: Missing")
            all_good = False
    
    for var in optional_vars:
        if os.getenv(var):
            print(f"✅ {var}: Set")
        else:
            print(f"⚠️ {var}: Not set (optional)")
    
    return all_good

def main():
    print("🔍 Titweng Backend System Check")
    print("=" * 40)
    
    results = {
        "database": check_database(),
        "ml_services": check_ml_services(),
        "environment": check_environment(),
        "api": check_api_endpoints()
    }
    
    print("\n📊 Summary:")
    print("=" * 40)
    
    for component, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {component.title()}: {'OK' if status else 'FAILED'}")
    
    all_critical_ok = results["database"] and results["ml_services"] and results["environment"]
    
    if all_critical_ok:
        print("\n🎉 All critical systems operational!")
        return 0
    else:
        print("\n⚠️ Some critical systems need attention!")
        return 1

if __name__ == "__main__":
    sys.exit(main())