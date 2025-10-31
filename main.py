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
from fastapi.staticfiles import StaticFiles

#from utils import detect_nose, preprocess_image, extract_embedding, generate_qr_code, generate_receipt_pdf, yolo_model, siamese_model


from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from contextlib import asynccontextmanager

# Database & Auth
from database import get_db
from models import Cow, Owner, Embedding, User, VerificationLog, Report
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
    from fastapi_mail import FastMail, MessageSchema, ConnectionConfig
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

# Import LOCAL ML client for trained Siamese model
from ml_client_local import ml_client

# ---------------------------
# FastAPI Lifespan
# ---------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs("static/qrcodes", exist_ok=True)
    os.makedirs("static/receipts", exist_ok=True)
    os.makedirs("static/cow_faces", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create tables and admin user on startup
    try:
        from database import engine, SessionLocal
        from models import Base, User
        import bcrypt
        
        # Install pgvector extension and create tables
        logger.info("Installing pgvector extension...")
        with engine.begin() as conn:
            from sqlalchemy import text
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        logger.info("✅ pgvector extension installed")
        
        logger.info("Creating database tables...")
        Base.metadata.create_all(bind=engine)  # Create tables if they don't exist
        logger.info("✅ Database setup completed")
        
        # Add facial_image_path column if it doesn't exist
        logger.info("Checking for facial_image_path column...")
        with engine.begin() as conn:
            try:
                conn.execute(text("ALTER TABLE cows ADD COLUMN facial_image_path TEXT;"))
                logger.info("✅ Added facial_image_path column")
            except Exception as e:
                if "already exists" in str(e) or "duplicate column" in str(e):
                    logger.info("✅ facial_image_path column already exists")
                else:
                    logger.warning(f"Column migration warning: {e}")
        
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
    
    logger.info("✅ Using local trained Siamese model")
    yield
    logger.info("Shutting down...")

app = FastAPI(title="Cattle Nose Recognition API", version="2.0", lifespan=lifespan)
# Using trained Siamese model

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

# Mount static files for serving images
app.mount("/static", StaticFiles(directory="static"), name="static")

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
        if embedding is not None and embedding.size > 0:
            logger.info(f"✅ Embedding extracted successfully: {embedding.shape}")
            return embedding
        logger.error("ML client returned None or empty embedding")
        raise HTTPException(status_code=503, detail="Failed to extract embedding")
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        raise HTTPException(status_code=503, detail="Embedding service unavailable")

def detect_nose(image_bytes: bytes) -> Optional[dict]:
    try:
        result = ml_client.detect_nose(image_bytes)
        if result:
            return result  # Return detection result with bbox and confidence
        return None
    except Exception as e:
        logger.warning(f"Nose detection failed: {e}")
        return None

# ---------------------------
# Email Configuration
# ---------------------------
# Use port 465 with SSL for better Gmail compatibility
mail_config = None
fastmail = None

if FASTMAIL_AVAILABLE:
    mail_config = ConnectionConfig(
        MAIL_USERNAME=os.getenv("SMTP_FROM_EMAIL", "g.bior@alustudent.com").replace("mailto:", ""),
        MAIL_PASSWORD=os.getenv("SMTP_PASSWORD", ""),
        MAIL_FROM=os.getenv("SMTP_FROM_EMAIL", "g.bior@alustudent.com").replace("mailto:", ""),
        MAIL_PORT=465,  # Use SSL port instead of STARTTLS
        MAIL_SERVER="smtp.gmail.com",
        MAIL_FROM_NAME=os.getenv("SMTP_FROM_NAME", "Titweng Cattle System"),
        MAIL_STARTTLS=False,
        MAIL_SSL_TLS=True,  # Use SSL instead of STARTTLS
        USE_CREDENTIALS=True,
        VALIDATE_CERTS=True
    )
    fastmail = FastMail(mail_config)

async def send_email(recipient_email: str, subject: str, body: str, pdf_path: Optional[str] = None):
    if not FASTMAIL_AVAILABLE or not fastmail:
        logger.warning("Email service not available")
        return False
    
    try:
        message = MessageSchema(
            subject=subject,
            recipients=[recipient_email],
            body=body,
            subtype="html"
        )
        
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                message.attachments = [{
                    "file": f.read(),
                    "filename": "registration_receipt.pdf",
                    "content_type": "application/pdf"
                }]
        
        await fastmail.send_message(message)
        logger.info(f"Email sent successfully to {recipient_email}")
        return True
    except Exception as e:
        logger.error(f"Failed to send email to {recipient_email}: {e}")
        return False

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
    
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or user.role != "admin":
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not bcrypt.checkpw(form_data.password.encode('utf-8'), user.password_hash.encode('utf-8')):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.username, "role": user.role})
    return {"access_token": token, "token_type": "bearer"}

# ---------------------------
# Email Sending Functions
# ---------------------------
async def send_registration_email(to_email: str, owner_name: str, cow, pdf_path: str):
    """Send registration confirmation email via Brevo API"""
    try:
        import requests
        import base64
        
        brevo_api_key = os.getenv("BREVO_API_KEY")
        sender_email = os.getenv("SMTP_FROM_EMAIL", "g.bior@alustudent.com")
        sender_name = os.getenv("SMTP_FROM_NAME", "Titweng Cattle System")
        
        if not brevo_api_key:
            logger.warning("Brevo API key not configured, logging email instead")
            logger.info(f"📧 Registration email for {to_email} - cow {cow.cow_tag}")
            return True
        
        # Prepare attachment
        attachment = None
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_content = base64.b64encode(f.read()).decode()
                attachment = {
                    "content": pdf_content,
                    "name": f"Titweng_Certificate_{cow.cow_tag}.pdf"
                }
        
        # Send via Brevo API
        url = "https://api.brevo.com/v3/smtp/email"
        headers = {
            "api-key": brevo_api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "sender": {"name": sender_name, "email": sender_email},
            "to": [{"email": to_email, "name": owner_name}],
            "subject": f"🐄 Cow Registration Successful - {cow.cow_tag}",
            "htmlContent": f"""
            <h2>🎉 Cow Registration Successful!</h2>
            <p>Dear <strong>{owner_name}</strong>,</p>
            <p>Your cow has been successfully registered!</p>
            <ul>
            <li><strong>Cow Tag:</strong> {cow.cow_tag}</li>
            <li><strong>Breed:</strong> {cow.breed or 'N/A'}</li>
            <li><strong>Color:</strong> {cow.color or 'N/A'}</li>
            <li><strong>Age:</strong> {cow.age or 'N/A'} years</li>
            <li><strong>Registration Date:</strong> {cow.registration_date.strftime('%Y-%m-%d')}</li>
            </ul>
            <p>Best regards,<br>Titweng Team</p>
            """
        }
        
        if attachment:
            data["attachment"] = [attachment]
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 201:
            logger.info(f"✅ Registration email sent via Brevo to {to_email}")
            return True
        else:
            logger.error(f"Brevo API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send email via Brevo: {e}")
        return False

async def send_transfer_email(to_email: str, new_owner_name: str, old_owner_name: str, cow, pdf_path: str):
    """Send ownership transfer confirmation email via Brevo API"""
    try:
        import requests
        import base64
        
        brevo_api_key = os.getenv("BREVO_API_KEY")
        sender_email = os.getenv("SMTP_FROM_EMAIL", "g.bior@alustudent.com")
        sender_name = os.getenv("SMTP_FROM_NAME", "Titweng Cattle System")
        
        if not brevo_api_key:
            logger.warning("Brevo API key not configured, logging email instead")
            logger.info(f"📧 Transfer email for {to_email} - cow {cow.cow_tag}")
            return True
        
        # Prepare attachment
        attachment = None
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as f:
                pdf_content = base64.b64encode(f.read()).decode()
                attachment = {
                    "content": pdf_content,
                    "name": f"Titweng_Transfer_{cow.cow_tag}.pdf"
                }
        
        # Send via Brevo API
        url = "https://api.brevo.com/v3/smtp/email"
        headers = {
            "api-key": brevo_api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "sender": {"name": sender_name, "email": sender_email},
            "to": [{"email": to_email, "name": new_owner_name}],
            "subject": f"🔄 Cow Ownership Transfer - {cow.cow_tag}",
            "htmlContent": f"""
            <h2>🔄 Ownership Transfer Completed!</h2>
            <p>Dear <strong>{new_owner_name}</strong>,</p>
            <p>Cow <strong>{cow.cow_tag}</strong> has been transferred to you!</p>
            <ul>
            <li><strong>Cow Tag:</strong> {cow.cow_tag}</li>
            <li><strong>Previous Owner:</strong> {old_owner_name}</li>
            <li><strong>New Owner:</strong> {new_owner_name}</li>
            <li><strong>Transfer Date:</strong> {cow.transfer_date.strftime('%Y-%m-%d')}</li>
            </ul>
            <p>Best regards,<br>Titweng Team</p>
            """
        }
        
        if attachment:
            data["attachment"] = [attachment]
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 201:
            logger.info(f"✅ Transfer email sent via Brevo to {to_email}")
            return True
        else:
            logger.error(f"Brevo API error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send transfer email via Brevo: {e}")
        return False

# ---------------------------
# ML Model Diagnostics
# ---------------------------
@app.post("/test-ml-models", summary="Test ML model outputs and dimensions")
async def test_ml_models(
    file: UploadFile = File(...)
):
    """Diagnostic endpoint to test ML model outputs"""
    try:
        contents = await file.read()
        logger.info(f"Testing ML models with file: {file.filename}, size: {len(contents)} bytes")
        
        # Test YOLO detector (returns dummy result since images are pre-cropped)
        yolo_result = detect_nose(contents)
        
        # Test Siamese embedder directly with ml_client (not through extract_embedding wrapper)
        embedding = ml_client.extract_embedding(contents)
        
        # EXACT same output format as test_model_outputs.py
        nose_detected = bool(yolo_result.get("detected", False)) if yolo_result else False
        bbox = yolo_result.get("bbox", []) if yolo_result else []
        confidence = float(yolo_result.get("confidence", 0.0)) if yolo_result else 0.0
        
        embedding_generated = bool(embedding is not None and embedding.size > 0)
        embedding_dim = int(embedding.shape[0]) if embedding is not None and embedding.size > 0 else 0
        embedding_norm = float(np.linalg.norm(embedding)) if embedding is not None and embedding.size > 0 else 0.0
        embedding_normalized = bool(np.isclose(embedding_norm, 1.0)) if embedding is not None and embedding.size > 0 else False
        
        dimension_match = embedding_dim == 256
        pgvector_format = "[" + ",".join(map(str, embedding.tolist())) + "]" if embedding is not None and embedding.size > 0 else []
        
        return {
            "success": embedding_generated,
            "yolo_detector": {
                "nose_detected": nose_detected,
                "bbox": bbox,
                "confidence": confidence
            },
            "siamese_embedder": {
                "embedding_generated": embedding_generated,
                "embedding_dimension": embedding_dim,
                "embedding_type": str(type(embedding)) if embedding is not None else "None",
                "embedding_sample": embedding[:5].tolist() if embedding is not None and embedding.size > 0 else [],
                "embedding_norm": round(embedding_norm, 6),
                "embedding_normalized": embedding_normalized
            },
            "database_compatibility": {
                "expected_dimension": 256,
                "dimension_match": dimension_match,
                "pgvector_format": pgvector_format
            }
        }
            
    except Exception as e:
        logger.error(f"ML model test failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "debug_info": {
                "file_size": len(contents) if 'contents' in locals() else 0,
                "file_type": file.content_type if file else "unknown"
            }
        }



# ---------------------------
# Health Check
# ---------------------------
@app.get("/health")
def health_check(db: Session = Depends(get_db)):
    health_status = {
        "status": "healthy",
        "database": "unknown",
        "ml_services": {
            "siamese_api": "unknown"
        },
        "email_available": FASTMAIL_AVAILABLE,
        "endpoints": {
            "admin": "/admin/*",
            "mobile": "/mobile/*"
        }
    }
    
    # Test database connection
    try:
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        health_status["database"] = "connected"
    except Exception as e:
        health_status["database"] = f"failed: {str(e)}"
        health_status["status"] = "unhealthy"
    
    # Test local ML model
    try:
        # Test if model can load and process
        test_embedding = ml_client.extract_embedding(b"dummy_test")
        if test_embedding is not None:
            health_status["ml_services"]["siamese_api"] = "local_model_ready"
        else:
            health_status["ml_services"]["siamese_api"] = "local_model_failed"
            health_status["status"] = "degraded"
    except Exception as e:
        health_status["ml_services"]["siamese_api"] = f"local_model_error: {str(e)}"
        health_status["status"] = "degraded"
    
    return health_status

# ---------------------------
# Email Test Endpoint
# ---------------------------
@app.get("/test-email-config")
def test_email_config():
    """Test email configuration and credentials"""
    config_status = {
        "fastmail_available": FASTMAIL_AVAILABLE,
        "sender_email": os.getenv("SMTP_FROM_EMAIL", "NOT_SET").replace("mailto:", ""),
        "sender_password_set": bool(os.getenv("SMTP_PASSWORD")),
        "smtp_server": os.getenv("SMTP_SERVER", "NOT_SET"),
        "smtp_port": "465 (SSL)",
        "sender_name": os.getenv("SMTP_FROM_NAME", "NOT_SET")
    }
    
    # Test SMTP connection with SSL
    try:
        import smtplib
        sender_email = config_status["sender_email"]
        sender_password = os.getenv("SMTP_PASSWORD", "")
        
        if sender_email and sender_password:
            server = smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30)
            server.login(sender_email, sender_password)
            server.quit()
            config_status["smtp_connection"] = "success"
        else:
            config_status["smtp_connection"] = "credentials_missing"
            
    except Exception as e:
        config_status["smtp_connection"] = f"failed: {str(e)}"
    
    
    return config_status


