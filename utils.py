import os
from typing import Optional, Dict, Any
import numpy as np
from PIL import Image
# Removed torch dependencies - using HF APIs

# Optional: QR + PDF
try:
    import qrcode
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.pdfgen import canvas
    QR_AVAILABLE = True
except Exception:
    QR_AVAILABLE = False

# Removed Siamese Network class - using HF APIs

# Import LOCAL ML client for trained Siamese model
from ml_client_local import ml_client

# Removed image preprocessing - using HF APIs

def extract_embedding(image_bytes: bytes) -> Optional[np.ndarray]:
    """Extract embedding using external ML API"""
    return ml_client.extract_embedding(image_bytes)

def detect_nose(image_bytes: bytes) -> Optional[dict]:
    """Detect nose using external YOLO API"""
    return ml_client.detect_nose(image_bytes)

# ---------------------------
# Matching helpers
# ---------------------------
def safe_duplicate_check(query_emb: np.ndarray, embeddings_db: dict, threshold: float = 0.2):
    best = None
    best_dist = float('inf')
    for cid, emb in embeddings_db.items():
        dist = np.linalg.norm(query_emb - emb)
        if dist < best_dist:
            best_dist = dist
            best = {"cow_id": cid, "distance": dist}
    return {"is_duplicate": best_dist < threshold, "best_match": best}

def find_top_matches(query_emb: np.ndarray, embeddings_db: dict, top_k: int = 3):
    matches = []
    for cid, emb in embeddings_db.items():
        dist = np.linalg.norm(query_emb - emb)
        matches.append({"cow_id": cid, "distance": dist})
    matches.sort(key=lambda x: x["distance"])
    return matches[:top_k]

# ---------------------------
# QR / PDF placeholders
# ---------------------------
def generate_qr_code(data: str, save_path: str):
    if QR_AVAILABLE:
        img = qrcode.make(data)
        img.save(save_path)
        return save_path
    return None

def generate_transfer_receipt_pdf(cow_data: Dict[str,Any], old_owner_data: Dict[str,Any], new_owner_data: Dict[str,Any], qr_path: Optional[str], save_path: str):
    if QR_AVAILABLE:
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from datetime import datetime
        
        c = canvas.Canvas(save_path, pagesize=A4)
        width, height = A4
        
        # Header with logo (centered)
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            logo_width = 120
            logo_x = (width - logo_width) / 2
            c.drawImage(logo_path, logo_x, height-100, width=logo_width, height=60, mask='auto')
        
        # Title (centered)
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(colors.darkred)
        title = "CATTLE OWNERSHIP TRANSFER CERTIFICATE"
        title_width = c.stringWidth(title, "Helvetica-Bold", 20)
        c.drawString((width - title_width) / 2, height-130, title)
        
        # Subtitle
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.black)
        subtitle = "Official Ownership Transfer Document"
        subtitle_width = c.stringWidth(subtitle, "Helvetica", 12)
        c.drawString((width - subtitle_width) / 2, height-150, subtitle)
        
        # Transfer Notice (centered and prominent)
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.darkred)
        transfer_notice = f"OWNERSHIP TRANSFERRED FROM {old_owner_data.get('full_name', 'N/A').upper()} TO {new_owner_data.get('full_name', 'N/A').upper()}"
        notice_width = c.stringWidth(transfer_notice, "Helvetica-Bold", 14)
        if notice_width > width - 80:  # If too long, use smaller font
            c.setFont("Helvetica-Bold", 12)
            notice_width = c.stringWidth(transfer_notice, "Helvetica-Bold", 12)
        c.drawString((width - notice_width) / 2, height-175, transfer_notice)
        
        # Border
        c.setStrokeColor(colors.darkred)
        c.setLineWidth(2)
        c.rect(40, 40, width-80, height-80)
        
        # Transfer Information
        y = height - 220
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.darkred)
        c.drawString(60, y, "OWNERSHIP TRANSFER DETAILS")
        c.line(60, y-5, 300, y-5)
        
        y -= 30
        c.setFont("Helvetica", 11)
        c.setFillColor(colors.black)
        
        transfer_info = [
            ("Cow ID:", cow_data.get("cow_id", "N/A")),
            ("Cow Tag:", cow_data.get("cow_tag", "N/A")),
            ("Transfer Date:", cow_data.get("transfer_date", "N/A"))
        ]
        
        for label, value in transfer_info:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(60, y, label)
            c.setFont("Helvetica", 11)
            c.drawString(180, y, str(value))
            y -= 20
        
        # Previous Owner Section
        y -= 20
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.darkred)
        c.drawString(60, y, "PREVIOUS OWNER")
        c.line(60, y-5, 200, y-5)
        
        y -= 30
        c.setFillColor(colors.black)
        
        old_owner_info = [
            ("Name:", old_owner_data.get("full_name", "N/A")),
            ("Email:", old_owner_data.get("email", "N/A")),
            ("Phone:", old_owner_data.get("phone", "N/A"))
        ]
        
        for label, value in old_owner_info:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(60, y, label)
            c.setFont("Helvetica", 11)
            c.drawString(180, y, str(value))
            y -= 20
        
        # New Owner Section
        y -= 20
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.darkgreen)
        c.drawString(60, y, "NEW OWNER")
        c.line(60, y-5, 180, y-5)
        
        y -= 30
        c.setFillColor(colors.black)
        
        new_owner_info = [
            ("Name:", new_owner_data.get("full_name", "N/A")),
            ("Email:", new_owner_data.get("email", "N/A")),
            ("Phone:", new_owner_data.get("phone", "N/A")),
            ("Address:", new_owner_data.get("address", "N/A")),
            ("National ID:", new_owner_data.get("national_id", "N/A"))
        ]
        
        for label, value in new_owner_info:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(60, y, label)
            c.setFont("Helvetica", 11)
            c.drawString(180, y, str(value))
            y -= 20
        
        # Footer with signature and QR code
        footer_y = 120
        
        # Signature (left side)
        signature_path = "assets/signature.png"
        if os.path.exists(signature_path):
            c.drawImage(signature_path, 60, footer_y, width=120, height=40, mask='auto')
            c.setFont("Helvetica", 9)
            c.drawString(60, footer_y-10, "Titweng Administrator")
        
        # QR Code (right side)
        if qr_path and os.path.exists(qr_path):
            qr_x = width - 140
            c.drawImage(qr_path, qr_x, footer_y, width=80, height=80)
            c.setFont("Helvetica", 8)
            c.drawString(qr_x, footer_y-10, "Scan for verification")
        
        # Transfer certificate info
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        cert_info = f"Transfer Certificate No: TW-T-{cow_data.get('cow_id', '000')}-{datetime.now().year}"
        c.drawString(60, 60, cert_info)
        c.drawString(60, 50, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Important notice
        c.setFont("Helvetica-Bold", 9)
        c.setFillColor(colors.darkred)
        c.drawString(60, 80, "IMPORTANT: Original registration receipt is NO LONGER VALID")
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        c.drawString(60, 70, "This transfer certificate is the ONLY valid ownership document")
        
        c.save()
        return save_path
    return None

def generate_receipt_pdf(cow_data: Dict[str,Any], owner_data: Dict[str,Any], qr_path: Optional[str], save_path: str):
    if QR_AVAILABLE:
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import inch
        from datetime import datetime
        
        c = canvas.Canvas(save_path, pagesize=A4)
        width, height = A4
        
        # Header with logo (centered)
        logo_path = "assets/logo.png"
        if os.path.exists(logo_path):
            logo_width = 120
            logo_x = (width - logo_width) / 2  # Center horizontally
            c.drawImage(logo_path, logo_x, height-100, width=logo_width, height=60, mask='auto')
        
        # Title (centered)
        c.setFont("Helvetica-Bold", 20)
        c.setFillColor(colors.darkblue)
        title = "CATTLE REGISTRATION CERTIFICATE"
        title_width = c.stringWidth(title, "Helvetica-Bold", 20)
        c.drawString((width - title_width) / 2, height-130, title)
        
        # Subtitle
        c.setFont("Helvetica", 12)
        c.setFillColor(colors.black)
        subtitle = "Official Registration Document - Original Owner"
        subtitle_width = c.stringWidth(subtitle, "Helvetica", 12)
        c.drawString((width - subtitle_width) / 2, height-150, subtitle)
        
        # Registration Notice (centered)
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.darkblue)
        reg_notice = f"REGISTERED TO: {owner_data.get('full_name', 'N/A').upper()}"
        notice_width = c.stringWidth(reg_notice, "Helvetica-Bold", 14)
        c.drawString((width - notice_width) / 2, height-175, reg_notice)
        
        # Border
        c.setStrokeColor(colors.darkblue)
        c.setLineWidth(2)
        c.rect(40, 40, width-80, height-80)
        
        # Cow Information Section
        y = height - 220
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.darkblue)
        c.drawString(60, y, "CATTLE INFORMATION")
        c.setLineWidth(1)
        c.line(60, y-5, 280, y-5)
        
        y -= 30
        c.setFont("Helvetica", 11)
        c.setFillColor(colors.black)
        
        cow_info = [
            ("Cow ID:", cow_data.get("cow_id", "N/A")),
            ("Cow Tag:", cow_data.get("cow_tag", "N/A")),
            ("Breed:", cow_data.get("breed", "N/A")),
            ("Color:", cow_data.get("color", "N/A")),
            ("Age:", f"{cow_data.get('age', 'N/A')} years" if cow_data.get('age') else "N/A"),
            ("Registration Date:", cow_data.get("registration_date", "N/A"))
        ]
        
        for label, value in cow_info:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(60, y, label)
            c.setFont("Helvetica", 11)
            c.drawString(180, y, str(value))
            y -= 20
        
        # Owner Information Section
        y -= 20
        c.setFont("Helvetica-Bold", 14)
        c.setFillColor(colors.darkblue)
        c.drawString(60, y, "OWNER INFORMATION")
        c.line(60, y-5, 280, y-5)
        
        y -= 30
        c.setFillColor(colors.black)
        
        owner_info = [
            ("Full Name:", owner_data.get("full_name", "N/A")),
            ("Email:", owner_data.get("email", "N/A")),
            ("Phone:", owner_data.get("phone", "N/A")),
            ("Address:", owner_data.get("address", "N/A")),
            ("National ID:", owner_data.get("national_id", "N/A"))
        ]
        
        for label, value in owner_info:
            c.setFont("Helvetica-Bold", 11)
            c.drawString(60, y, label)
            c.setFont("Helvetica", 11)
            c.drawString(180, y, str(value))
            y -= 20
        
        # Footer with signature and QR code (same line)
        footer_y = 120
        
        # Signature (left side)
        signature_path = "assets/signature.png"
        if os.path.exists(signature_path):
            c.drawImage(signature_path, 60, footer_y, width=120, height=40, mask='auto')
            c.setFont("Helvetica", 9)
            c.drawString(60, footer_y-10, "Titweng Administrator")
        
        # QR Code (right side)
        if qr_path and os.path.exists(qr_path):
            qr_x = width - 140  # Right aligned
            c.drawImage(qr_path, qr_x, footer_y, width=80, height=80)
            c.setFont("Helvetica", 8)
            c.drawString(qr_x, footer_y-10, "Scan for verification")
        
        # Certificate number and date
        c.setFont("Helvetica", 8)
        c.setFillColor(colors.grey)
        cert_info = f"Certificate No: TW-{cow_data.get('cow_id', '000')}-{datetime.now().year}"
        c.drawString(60, 60, cert_info)
        c.drawString(60, 50, f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        c.save()
        return save_path
    return None

# ---------------------------
# Email Functions
# ---------------------------
def send_transfer_email(to_email: str, new_owner_name: str, old_owner_name: str, cow, pdf_path: str):
    """Send ownership transfer confirmation email with PDF attachment"""
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        import os
        from email_config import EMAIL_CONFIG
        
        # Email configuration
        smtp_server = EMAIL_CONFIG["smtp_server"]
        smtp_port = EMAIL_CONFIG["smtp_port"]
        sender_email = EMAIL_CONFIG["sender_email"]
        sender_password = EMAIL_CONFIG["sender_password"]
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{sender_email}>"
        msg['To'] = to_email
        msg['Subject'] = f"Cow Ownership Transfer Confirmation - {cow.cow_tag}"
        
        # Email body
        body = f"""
        Dear {new_owner_name},
        
        Congratulations! The ownership of cow {cow.cow_tag} has been successfully transferred to you.
        
        Transfer Details:
        - Cow Tag: {cow.cow_tag}
        - Breed: {cow.breed or 'N/A'}
        - Color: {cow.color or 'N/A'}
        - Age: {cow.age or 'N/A'} years
        - Previous Owner: {old_owner_name}
        - Transfer Date: {cow.transfer_date.strftime('%Y-%m-%d')}
        
        Please find the ownership transfer certificate attached to this email.
        
        IMPORTANT: The original registration receipt is no longer valid. 
        This transfer certificate is now your official ownership document.
        
        Best regards,
        Titweng Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF if exists
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= cow_{cow.cow_id}_transfer_certificate.pdf'
                )
                msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Transfer email sent to {to_email} for cow {cow.cow_id}")
        
    except Exception as e:
        print(f"❌ Failed to send transfer email: {e}")

def send_registration_email(to_email: str, owner_name: str, cow, pdf_path: str):
    """Send registration confirmation email with PDF attachment"""
    try:
        import smtplib
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText
        from email.mime.base import MIMEBase
        from email import encoders
        import os
        from email_config import EMAIL_CONFIG
        
        # Email configuration
        smtp_server = EMAIL_CONFIG["smtp_server"]
        smtp_port = EMAIL_CONFIG["smtp_port"]
        sender_email = EMAIL_CONFIG["sender_email"]
        sender_password = EMAIL_CONFIG["sender_password"]
        
        # Create message
        msg = MIMEMultipart()
        msg['From'] = f"{EMAIL_CONFIG['sender_name']} <{sender_email}>"
        msg['To'] = to_email
        msg['Subject'] = f"Cow Registration Confirmation - {cow.cow_tag}"
        
        # Email body
        body = f"""
        Dear {owner_name},
        
        Your cow has been successfully registered in the Titweng system!
        
        Cow Details:
        - Cow Tag: {cow.cow_tag}
        - Breed: {cow.breed or 'N/A'}
        - Color: {cow.color or 'N/A'}
        - Age: {cow.age or 'N/A'} years
        - Registration Date: {cow.registration_date.strftime('%Y-%m-%d')}
        
        Please find the registration receipt attached to this email.
        
        Best regards,
        Titweng Team
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Attach PDF if exists
        if pdf_path and os.path.exists(pdf_path):
            with open(pdf_path, "rb") as attachment:
                part = MIMEBase('application', 'octet-stream')
                part.set_payload(attachment.read())
                encoders.encode_base64(part)
                part.add_header(
                    'Content-Disposition',
                    f'attachment; filename= cow_{cow.cow_id}_receipt.pdf'
                )
                msg.attach(part)
        
        # Send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(sender_email, sender_password)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Registration email sent to {to_email} for cow {cow.cow_id}")
        
    except Exception as e:
        print(f"❌ Failed to send email: {e}")

def send_sms_registration(phone_number: str, cow_id: int, qr_link: str):
    print(f"Sending SMS to {phone_number} for cow {cow_id} with QR {qr_link}")

# ---------------------------
# Verification Email/SMS placeholders
# ---------------------------
def send_email_verification(to_email: str, owner_name: str, verification_code: str):
    print(f"Sending verification email to {to_email} for {owner_name} with code {verification_code}")

def send_sms_verification(phone_number: str, verification_code: str):
    print(f"Sending verification SMS to {phone_number} with code {verification_code}")


# utils.py
from passlib.context import CryptContext

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    # truncate to 72 chars for bcrypt
    return pwd_context.hash(password[:72])

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

