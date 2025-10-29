from sqlalchemy import Column, Integer, String, Text, ForeignKey, TIMESTAMP, Float
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector
from database import Base

# 1️⃣ Owners Table
class Owner(Base):
    __tablename__ = "owners"

    owner_id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String(100), nullable=False)
    email = Column(String(120))
    phone = Column(String(20))
    address = Column(Text)
    national_id = Column(String(50))
    created_at = Column(TIMESTAMP, server_default=func.now())

    cows = relationship("Cow", back_populates="owner")


# 2️⃣ Cows Table
class Cow(Base):
    __tablename__ = "cows"

    cow_id = Column(Integer, primary_key=True, index=True)
    cow_tag = Column(String(50), unique=True, nullable=False)
    breed = Column(String(100), nullable=False)
    color = Column(String(50), nullable=False)
    age = Column(Integer, nullable=False)
    registration_date = Column(TIMESTAMP, server_default=func.now())
    owner_id = Column(Integer, ForeignKey("owners.owner_id", ondelete="CASCADE"))
    qr_code_path = Column(Text)  # Path to QR code file
    receipt_pdf_path = Column(Text)  # Path to PDF receipt file
    qr_code_data = Column(Text)  # QR code verification URL
    transfer_receipt_path = Column(Text)  # Path to transfer receipt
    ownership_status = Column(String(20), default="original")  # original, transferred
    transfer_date = Column(TIMESTAMP)  # Date of ownership transfer
    previous_owner_id = Column(Integer)  # Previous owner for transfer history
    facial_image_path = Column(Text)  # Path to cow facial image for verification

    owner = relationship("Owner", back_populates="cows")
    embeddings = relationship("Embedding", back_populates="cow")
    verification_logs = relationship("VerificationLog", back_populates="cow")


# 3️⃣ Embeddings Table
class Embedding(Base):
    __tablename__ = "embeddings"

    embedding_id = Column(Integer, primary_key=True, index=True)
    cow_id = Column(Integer, ForeignKey("cows.cow_id", ondelete="CASCADE"))
    image_path = Column(Text)
    embedding = Column(Vector(256))  # Changed from 512 to 256
    image_angle = Column(String(20))  # 'front', 'left', 'right'
    quality_score = Column(Float)  # Image quality 0-1
    is_primary = Column(String(10), default="no")  # Mark best embedding
    created_at = Column(TIMESTAMP, server_default=func.now())

    cow = relationship("Cow", back_populates="embeddings")


# 4️⃣ Users Table (Admin Only)
class User(Base):
    __tablename__ = "users"

    user_id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), unique=True, nullable=False)
    email = Column(String(120), unique=True)
    phone = Column(String(20))
    password_hash = Column(Text, nullable=False)
    role = Column(String(20), default="admin")  # admin only
    user_type = Column(String(20), default="admin")
    created_at = Column(TIMESTAMP, server_default=func.now())


# 5️⃣ Verification Logs Table
class VerificationLog(Base):
    __tablename__ = "verification_logs"

    log_id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.user_id"), nullable=True)  # Optional for mobile verifications
    cow_id = Column(Integer, ForeignKey("cows.cow_id"))
    verification_image = Column(Text)
    similarity_score = Column(Float)
    location = Column(Text)
    verified = Column(String(20))
    created_at = Column(TIMESTAMP, server_default=func.now())

    user = relationship("User")
    cow = relationship("Cow", back_populates="verification_logs")


# 6️⃣ Reports Table
class Report(Base):
    __tablename__ = "reports"

    report_id = Column(Integer, primary_key=True, index=True)
    reporter_name = Column(String(100), nullable=False)
    reporter_phone = Column(String(20))
    reporter_email = Column(String(120))
    cow_tag = Column(String(50))
    report_type = Column(String(50), nullable=False)  # suspect, theft, other
    subject = Column(String(200), nullable=False)
    message = Column(Text, nullable=False)
    location = Column(Text)
    status = Column(String(20), default="pending")  # pending, resolved, closed
    admin_reply = Column(Text)
    created_at = Column(TIMESTAMP, server_default=func.now())
    updated_at = Column(TIMESTAMP, server_default=func.now(), onupdate=func.now())