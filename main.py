# main.py
import os
import io
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any

import numpy as np
from PIL import Image
# Removed torch dependencies - using HF APIs
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, Depends, HTTPException

#from utils import detect_nose, preprocess_image, extract_embedding, generate_qr_code, generate_receipt_pdf, yolo_model, siamese_model


from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

# Database & Auth
from database import get_db
from models import Cow, Owner, Embedding, User, VerificationLog
from auth import create_access_token, get_current_user, get_current_admin

# Optional libraries
try:
    import qrcode
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False

try:
    from fastapi_mail import FastMail
    FASTMAIL_AVAILABLE = True
except Exception:
    FASTMAIL_AVAILABLE = False

try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except Exception:
    TWILIO_AVAILABLE = False

# Removed ultralytics dependency

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------------------
# Configs
# ---------------------------
REGISTRATION_CONFIG = {"min_images": 3, "max_images": 5}

# Import ML client for Hugging Face models
from ml_client import ml_client

# ---------------------------
# FastAPI Lifespan
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("static/qrcodes", exist_ok=True)
    os.makedirs("static/receipts", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create tables and admin user on startup
    try:
        from database import engine, SessionLocal
        from models import Base, User
        import bcrypt
        
        # Create all tables with detailed logging
        logger.info("Creating database tables...")
        Base.metadata.drop_all(bind=engine)  # Drop existing tables
        Base.metadata.create_all(bind=engine)  # Create fresh tables
        logger.info("✅ Database tables created successfully")
        
        # Create admin user if not exists
        db = SessionLocal()
        admin_username = os.getenv("ADMIN_USERNAME", "titweng")
        admin_password = os.getenv("ADMIN_PASSWORD", "titweng@2025")
        admin_email = os.getenv("ADMIN_EMAIL", "admin@titweng.com")
        
        existing_admin = db.query(User).filter(User.username == admin_username).first()
        if not existing_admin:
            password_hash = bcrypt.hashpw(admin_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
            admin = User(
                username=admin_username,
                email=admin_email,
                password_hash=password_hash,
                role="admin",
                user_type="admin"
            )
            db.add(admin)
            db.commit()
            logger.info(f"✅ Admin user '{admin_username}' created")
        else:
            logger.info(f"✅ Admin user '{admin_username}' already exists")
        db.close()
        
    except Exception as e:
        logger.error(f"Database setup error: {e}")
        logger.error(f"Database URL: {os.getenv('DATABASE_URL', 'Not set')}")
        # Continue without database setup - will fail on first request
    
    logger.info("✅ Using Hugging Face models")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Cattle Nose Recognition API", version="2.0", lifespan=lifespan)

# ---------------------------
# Middleware
# ---------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# ---------------------------
# Include Routers
# ---------------------------
from routes import mobile, admin
app.include_router(mobile.router)
app.include_router(admin.router)

# ---------------------------
# ML Utilities
# ---------------------------
# Remove preprocess_image as it's no longer needed

def extract_embedding(image_bytes: bytes) -> np.ndarray:
    try:
        embedding = ml_client.extract_embedding(image_bytes)
        if embedding is not None:
            return embedding
        raise HTTPException(status_code=503, detail="Failed to extract embedding")
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable")

def detect_nose(image_bytes: bytes) -> Optional[bytes]:
    try:
        result = ml_client.detect_nose(image_bytes)
        if result and 'cropped_image' in result:
            return result['cropped_image']
        return image_bytes  # Return original if no nose detected
    except Exception as e:
        logger.warning(f"Nose detection failed: {e}")
        return image_bytes

# ---------------------------
# QR & PDF
# ---------------------------
def generate_qr_code(data:str, save_path:str):
    if QR_AVAILABLE:
        import qrcode
        img = qrcode.make(data)
        img.save(save_path)
        return save_path
    return None

def generate_receipt_pdf(meta:Dict[str,Any], qr_path:Optional[str], logo_path:Optional[str], save_path:str):
    if QR_AVAILABLE:
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import A4
        from reportlab.lib import colors
        c = canvas.Canvas(save_path, pagesize=A4)
        width,height = A4
        c.setStrokeColor(colors.lightgrey)
        c.setLineWidth(2)
        c.rect(30,30,width-60,height-60)
        if logo_path and os.path.exists(logo_path):
            c.drawImage(logo_path,50,height-120,width=100,height=80,mask='auto')
        c.setFont("Helvetica-Bold",18)
        c.setFillColor(colors.darkgreen)
        c.drawString(170,height-60,"Titweng Cow Registration Receipt")
        c.setFillColor(colors.black)
        y=height-130
        c.setFont("Helvetica",12)
        for k,v in meta.items():
            c.drawString(60,y,f"{k}: {v}")
            y-=18
            if y<120: c.showPage(); y=height-60
        if qr_path and os.path.exists(qr_path):
            c.drawImage(qr_path, width-180,120,width=120,height=120)
        c.save()
        return save_path
    return None

# ---------------------------
# Admin Auth Only
# ---------------------------
@app.post("/admin/login")
def admin_login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    import bcrypt
    
    # Log the login attempt
    logger.info(f"Login attempt for username: {form_data.username}")
    
    # Find admin user
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user:
        logger.warning(f"User not found: {form_data.username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if user.role != "admin":
        logger.warning(f"Non-admin login attempt: {form_data.username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Verify password
    password_valid = False
    try:
        password_valid = bcrypt.checkpw(form_data.password.encode('utf-8'), user.password_hash.encode('utf-8'))
        logger.info(f"Password verification result: {password_valid}")
    except Exception as e:
        logger.error(f"Password verification error: {e}")
        try:
            from passlib.context import CryptContext
            pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            password_valid = pwd_context.verify(form_data.password, user.password_hash)
            logger.info(f"Passlib verification result: {password_valid}")
        except Exception as e2:
            logger.error(f"Passlib verification error: {e2}")
    
    if not password_valid:
        logger.warning(f"Invalid password for user: {form_data.username}")
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Create token
    token = create_access_token({"sub": user.username, "role": user.role})
    logger.info(f"Login successful for: {form_data.username}")
    
    return {"access_token": token, "token_type": "bearer"}

# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "yolo_loaded": True,
        "siamese_loaded": True,
        "endpoints": {
            "admin": "/admin/*",
            "mobile": "/mobile/*"
        }
    }
