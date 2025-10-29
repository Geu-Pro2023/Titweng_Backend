#!/usr/bin/env python3
"""
Create admin user for Azure deployment
"""
import psycopg2
import bcrypt

def create_admin():
    """Create admin user in Azure PostgreSQL"""
    
    conn_params = {
        'host': 'titweng-db-public.postgres.database.azure.com',
        'database': 'postgres',
        'user': 'titwengadmin',
        'password': 'SecureDB2025!',
        'sslmode': 'require'
    }
    
    # Password hashing
    password = "titweng@2025"
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Check if admin already exists
        cursor.execute("SELECT username FROM users WHERE username = 'titweng';")
        existing = cursor.fetchone()
        
        if existing:
            print("Admin user already exists")
        else:
            # Create admin user
            cursor.execute("""
                INSERT INTO users (username, email, phone, password_hash, role, user_type)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                'titweng',
                'admin@titweng.com',
                '+250792104851',
                hashed_password,
                'admin',
                'admin'
            ))
            
            conn.commit()
            print("✓ Admin user created successfully")
            print("Username: titweng")
            print("Password: titweng@2025")
            
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    create_admin()