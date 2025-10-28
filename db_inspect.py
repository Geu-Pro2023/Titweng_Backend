#!/usr/bin/env python3
import os
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.orm import sessionmaker

def inspect_database():
    """Inspect database structure and data"""
    try:
        DATABASE_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://titweng_user:titweng!.com@localhost:5432/titweng")
        if DATABASE_URL.startswith("postgres://"):
            DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)
        
        engine = create_engine(DATABASE_URL)
        inspector = inspect(engine)
        
        print("🔍 Database Structure Inspection")
        print("=" * 50)
        
        # List all tables
        tables = inspector.get_table_names()
        print(f"📋 Tables ({len(tables)}):")
        for table in tables:
            print(f"  - {table}")
        
        print("\n📊 Table Details:")
        print("-" * 30)
        
        # Inspect each table
        for table_name in tables:
            print(f"\n🗂️  Table: {table_name}")
            columns = inspector.get_columns(table_name)
            
            print("   Columns:")
            for col in columns:
                nullable = "NULL" if col['nullable'] else "NOT NULL"
                print(f"     - {col['name']}: {col['type']} {nullable}")
            
            # Count records
            with engine.connect() as conn:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
                count = result.scalar()
                print(f"   Records: {count}")
        
        print("\n🔍 Data Samples:")
        print("-" * 30)
        
        # Show sample data from key tables
        with engine.connect() as conn:
            # Check users table
            if 'users' in tables:
                print("\n👤 Users:")
                result = conn.execute(text("SELECT username, email, role FROM users LIMIT 5"))
                for row in result:
                    print(f"   - {row.username} ({row.email}) - {row.role}")
            
            # Check cows table
            if 'cows' in tables:
                print("\n🐄 Cows:")
                result = conn.execute(text("SELECT cow_tag, breed, color, registration_date FROM cows LIMIT 5"))
                for row in result:
                    print(f"   - {row.cow_tag}: {row.breed} ({row.color}) - {row.registration_date}")
            
            # Check embeddings table
            if 'embeddings' in tables:
                print("\n🧠 Embeddings:")
                result = conn.execute(text("SELECT cow_id, array_length(embedding, 1) as dim FROM embeddings LIMIT 5"))
                for row in result:
                    print(f"   - Cow ID {row.cow_id}: {row.dim} dimensions")
            
            # Check owners table
            if 'owners' in tables:
                print("\n👨‍🌾 Owners:")
                result = conn.execute(text("SELECT name, email, phone FROM owners LIMIT 5"))
                for row in result:
                    print(f"   - {row.name} ({row.email}) - {row.phone}")
        
        # Check pgvector extension
        print("\n🔧 Extensions:")
        result = conn.execute(text("SELECT extname FROM pg_extension WHERE extname = 'vector'"))
        if result.fetchone():
            print("   ✅ pgvector extension installed")
        else:
            print("   ❌ pgvector extension missing")
        
        print("\n✅ Database inspection complete!")
        return True
        
    except Exception as e:
        print(f"❌ Database inspection failed: {e}")
        return False

if __name__ == "__main__":
    inspect_database()