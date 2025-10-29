#!/usr/bin/env python3
"""
Enable pgvector extension on Azure PostgreSQL
"""
import psycopg2
import os

def enable_pgvector():
    """Enable pgvector extension as admin user"""
    
    # Connect as admin user
    conn_params = {
        'host': 'titweng-db-public.postgres.database.azure.com',
        'database': 'postgres',
        'user': 'titwengadmin',
        'password': 'SecureDB2025!',
        'sslmode': 'require'
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        cursor = conn.cursor()
        
        # Try to create extension
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        print("✓ pgvector extension enabled successfully")
        
        # Verify extension
        cursor.execute("SELECT extname FROM pg_extension WHERE extname = 'vector';")
        result = cursor.fetchone()
        if result:
            print("✓ pgvector extension verified")
        else:
            print("✗ pgvector extension not found")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Trying alternative approach...")
        
        # Try with different privileges
        try:
            cursor.execute("SELECT version();")
            version = cursor.fetchone()
            print(f"PostgreSQL version: {version[0]}")
            
            # Check available extensions
            cursor.execute("SELECT name FROM pg_available_extensions WHERE name = 'vector';")
            available = cursor.fetchone()
            if available:
                print("✓ pgvector is available")
            else:
                print("✗ pgvector not available in this Azure instance")
                
        except Exception as e2:
            print(f"Secondary error: {e2}")
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    enable_pgvector()