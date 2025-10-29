#!/usr/bin/env python3
"""
Setup script for Azure PostgreSQL database
"""
from database import DATABASE_URL
from models import Base
from sqlalchemy import create_engine, text

def setup_database():
    """Setup database with pgvector extension and tables"""
    engine = create_engine(DATABASE_URL)
    
    # Create pgvector extension
    with engine.connect() as conn:
        try:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
            print("✓ pgvector extension created")
        except Exception as e:
            print(f"Extension creation: {e}")
    
    # Create tables
    Base.metadata.create_all(bind=engine)
    print("✓ Database tables created")

if __name__ == "__main__":
    setup_database()