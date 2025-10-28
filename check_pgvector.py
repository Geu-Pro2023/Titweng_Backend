#!/usr/bin/env python3
"""
Check if pgvector is properly installed and working
"""
import os
from sqlalchemy import create_engine, text
from database import get_db, engine
from models import Embedding, Cow, Owner

def check_pgvector_setup():
    print("🔍 Checking pgvector setup...")
    
    # Test 1: Check if pgvector extension is installed
    try:
        with engine.connect() as conn:
            result = conn.execute(text("SELECT * FROM pg_extension WHERE extname = 'vector';"))
            extension = result.fetchone()
            if extension:
                print("✅ pgvector extension is installed")
            else:
                print("❌ pgvector extension NOT installed")
                # Try to install it
                try:
                    conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    conn.commit()
                    print("✅ pgvector extension installed successfully")
                except Exception as e:
                    print(f"❌ Failed to install pgvector: {e}")
    except Exception as e:
        print(f"❌ Database connection error: {e}")
        return
    
    # Test 2: Check if embeddings table exists with vector column
    try:
        with engine.connect() as conn:
            result = conn.execute(text("""
                SELECT column_name, data_type 
                FROM information_schema.columns 
                WHERE table_name = 'embeddings' AND column_name = 'embedding';
            """))
            column = result.fetchone()
            if column:
                print(f"✅ Embeddings table has vector column: {column.data_type}")
            else:
                print("❌ Embeddings table missing vector column")
    except Exception as e:
        print(f"❌ Table check error: {e}")
    
    # Test 3: Check existing data
    try:
        db = next(get_db())
        
        # Count cows
        cow_count = db.query(Cow).count()
        print(f"📊 Total cows in database: {cow_count}")
        
        # Count embeddings
        embedding_count = db.query(Embedding).count()
        print(f"📊 Total embeddings in database: {embedding_count}")
        
        # Show sample embeddings
        if embedding_count > 0:
            sample = db.query(Embedding).first()
            print(f"📝 Sample embedding ID: {sample.embedding_id}")
            print(f"📝 Sample cow ID: {sample.cow_id}")
            print(f"📝 Sample embedding type: {type(sample.embedding)}")
            if sample.embedding:
                print(f"📝 Sample embedding length: {len(sample.embedding) if hasattr(sample.embedding, '__len__') else 'N/A'}")
        
        db.close()
        
    except Exception as e:
        print(f"❌ Data check error: {e}")
    
    # Test 4: Test pgvector similarity query
    try:
        with engine.connect() as conn:
            # Create a test vector
            test_vector = "[" + ",".join([str(0.1)] * 256) + "]"
            
            result = conn.execute(text("""
                SELECT COUNT(*) as count
                FROM embeddings 
                WHERE embedding IS NOT NULL;
            """))
            
            valid_embeddings = result.fetchone().count
            print(f"📊 Valid embeddings (not null): {valid_embeddings}")
            
            if valid_embeddings > 0:
                # Test similarity query
                result = conn.execute(text("""
                    SELECT cow_id, (1 - (embedding <=> :test_vector)) as similarity
                    FROM embeddings 
                    WHERE embedding IS NOT NULL
                    ORDER BY embedding <=> :test_vector 
                    LIMIT 1;
                """), {"test_vector": test_vector})
                
                similarity_result = result.fetchone()
                if similarity_result:
                    print(f"✅ pgvector similarity query works!")
                    print(f"📝 Test similarity: {similarity_result.similarity:.3f}")
                else:
                    print("❌ pgvector similarity query failed")
            
    except Exception as e:
        print(f"❌ pgvector test error: {e}")
    
    print("\n" + "="*50)
    print("🏁 pgvector check completed!")

if __name__ == "__main__":
    check_pgvector_setup()