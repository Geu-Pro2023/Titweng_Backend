#!/usr/bin/env python3
"""
Check database tables in Azure PostgreSQL
"""
import psycopg2

def check_tables():
    """Check what tables exist in the database"""
    
    conn_params = {
        'host': 'titweng-db-public.postgres.database.azure.com',
        'database': 'postgres',
        'user': 'titwengadmin',
        'password': 'SecureDB2025!',
        'sslmode': 'require'
    }
    
    try:
        conn = psycopg2.connect(**conn_params)
        cursor = conn.cursor()
        
        # Check extensions
        print("=== EXTENSIONS ===")
        cursor.execute("SELECT extname FROM pg_extension;")
        extensions = cursor.fetchall()
        for ext in extensions:
            print(f"✓ {ext[0]}")
        
        # Check tables
        print("\n=== TABLES ===")
        cursor.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            ORDER BY table_name;
        """)
        tables = cursor.fetchall()
        
        if tables:
            for table in tables:
                print(f"✓ {table[0]}")
                
                # Show table structure
                cursor.execute(f"""
                    SELECT column_name, data_type, is_nullable
                    FROM information_schema.columns 
                    WHERE table_name = '{table[0]}'
                    ORDER BY ordinal_position;
                """)
                columns = cursor.fetchall()
                for col in columns:
                    print(f"  - {col[0]} ({col[1]}) {'NULL' if col[2] == 'YES' else 'NOT NULL'}")
                print()
        else:
            print("No tables found")
            
    except Exception as e:
        print(f"Error: {e}")
    
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    check_tables()