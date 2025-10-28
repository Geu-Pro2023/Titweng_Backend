# routes/admin.py - Admin Dashboard Endpoints
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime
import numpy as np
import os
# Removed torch dependency

import main
from database import get_db
from models import Cow, Embedding, Owner, VerificationLog, Report
from auth import get_current_admin
from utils import generate_qr_code, generate_receipt_pdf, generate_transfer_receipt_pdf, QR_AVAILABLE
from cow_tag_generator import generate_secure_cow_tag

router = APIRouter(
    prefix="/admin",
    tags=["Admin Dashboard"]
)

REGISTRATION_CONFIG = {"min_images": 5, "max_images": 5}
REQUIRED_ANGLES = ["front", "left", "right", "top", "front2"]

# ---------------------------
# 1. Register New Cow (Owner + Cow Combined)
# ---------------------------
@router.post("/register-cow", summary="Register new cow with owner")
async def admin_register_new_cow(
    background_tasks: BackgroundTasks,
    # Owner Details
    owner_full_name: str = Form(...),
    owner_email: Optional[str] = Form(None),
    owner_phone: Optional[str] = Form(None),
    owner_address: Optional[str] = Form(None),
    owner_national_id: Optional[str] = Form(None),
    # Cow Details
    breed: str = Form(...),
    color: str = Form(...),
    age: int = Form(...),
    # Nose Print Images (5 required)
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    # Create owner first
    owner = Owner(
        full_name=owner_full_name,
        email=owner_email,
        phone=owner_phone,
        address=owner_address,
        national_id=owner_national_id
    )
    db.add(owner)
    db.commit()
    db.refresh(owner)
    
    # Auto-generate unique cow tag
    cow_tag = generate_secure_cow_tag(db, owner.owner_id)

    # Validate nose print images
    if len(files) != REGISTRATION_CONFIG["min_images"]:
        raise HTTPException(status_code=400, detail=f"Exactly {REGISTRATION_CONFIG['min_images']} nose print images required")

    # Check for duplicate registration using pgvector
    from sqlalchemy import text
    
    for idx, f in enumerate(files):
        contents = await f.read()
            
        # Detect nose using HF API
        nose_result = main.detect_nose(contents)
        if nose_result is None:
            raise HTTPException(status_code=400, detail=f"No cow nose detected in {f.filename}")
        
        # Extract embedding for duplicate check
        emb = main.extract_embedding(contents)
        if emb is None:
            raise HTTPException(status_code=400, detail=f"Failed to extract embedding from {f.filename}")
        emb_str = '[' + ','.join(map(str, emb.tolist())) + ']'
        
        # Check for duplicates using pgvector (much faster)
        query = text("""
            SELECT e.cow_id, c.cow_tag, o.full_name, (1 - (e.embedding <=> :query_emb)) as similarity
            FROM embeddings e
            JOIN cows c ON e.cow_id = c.cow_id
            JOIN owners o ON c.owner_id = o.owner_id
            WHERE (1 - (e.embedding <=> :query_emb)) > 0.93
            ORDER BY e.embedding <=> :query_emb
            LIMIT 1
        """)
        
        duplicate = db.execute(query, {"query_emb": emb_str}).fetchone()
        
        if duplicate:
            raise HTTPException(
                status_code=409, 
                detail=f"🚫 COW ALREADY REGISTERED! This cow is already in the system as {duplicate.cow_tag} owned by {duplicate.full_name}. Similarity: {duplicate.similarity:.3f}"
            )
    
    # Process nose print images if no duplicates found
    embeddings_data = []
    for idx, f in enumerate(files):
        contents = await f.read()
        nose_result = main.detect_nose(contents)
        emb = main.extract_embedding(contents)
        if emb is None:
            raise HTTPException(status_code=400, detail=f"Failed to extract embedding from {f.filename}")
        
        embeddings_data.append({
            "embedding": emb.tolist(),
            "angle": REQUIRED_ANGLES[idx],
            "quality_score": 0.8,
            "is_primary": "yes" if idx == 0 else "no",
            "image_path": f.filename
        })
    
    # Create cow record
    cow = Cow(
        cow_tag=cow_tag,
        breed=breed,
        color=color,
        age=age,
        owner_id=owner.owner_id,
        registration_date=datetime.utcnow()
    )
    db.add(cow)
    db.commit()
    db.refresh(cow)

    # Save embeddings
    for emb_data in embeddings_data:
        e = Embedding(
            cow_id=cow.cow_id,
            embedding=emb_data["embedding"],
            image_angle=emb_data["angle"],
            quality_score=emb_data["quality_score"],
            is_primary=emb_data["is_primary"],
            image_path=emb_data["image_path"]
        )
        db.add(e)
    db.commit()

    # Generate QR + PDF and send email
    if QR_AVAILABLE:
        qr_link = f"https://titweng.app/verify/{cow.cow_id}"
        qr_path = f"static/qrcodes/{cow.cow_id}.png"
        pdf_path = f"static/receipts/{cow.cow_id}_receipt.pdf"
        
        generate_qr_code(qr_link, qr_path)
        
        # Prepare cow and owner data for PDF
        cow_data = {
            "cow_id": cow.cow_id,
            "cow_tag": cow_tag,
            "breed": breed,
            "color": color,
            "age": age,
            "registration_date": cow.registration_date.strftime("%Y-%m-%d")
        }
        
        owner_data = {
            "full_name": owner.full_name,
            "email": owner.email,
            "phone": owner.phone,
            "address": owner.address,
            "national_id": owner.national_id
        }
        
        generate_receipt_pdf(cow_data, owner_data, qr_path, pdf_path)
        
        # Update cow with QR and PDF paths
        cow.qr_code_path = qr_path
        cow.receipt_pdf_path = pdf_path
        cow.qr_code_data = qr_link
        db.commit()
        
        # Send email in background if owner has email
        if owner.email:
            background_tasks.add_task(main.send_registration_email, owner.email, owner.full_name, cow, pdf_path)

    return {
        "success": True,
        "owner_id": owner.owner_id,
        "owner_name": owner.full_name,
        "cow_id": cow.cow_id,
        "cow_tag": cow_tag,
        "message": "Cow and owner registered successfully",
        "nose_prints_processed": len(embeddings_data),
        "qr_code_generated": bool(cow.qr_code_path),
        "receipt_generated": bool(cow.receipt_pdf_path),
        "email_sent": bool(owner.email)
    }

# ---------------------------
# 2. Get All Owners
# ---------------------------
@router.get("/owners", summary="Get all owners")
def admin_get_owners(
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    owners = db.query(Owner).all()
    return {
        "total_owners": len(owners),
        "owners": [{
            "owner_id": o.owner_id,
            "full_name": o.full_name,
            "email": o.email,
            "phone": o.phone,
            "address": o.address,
            "national_id": o.national_id,
            "created_at": o.created_at
        } for o in owners]
    }

# ---------------------------
# 3. Get All Cows (Admin)
# ---------------------------
@router.get("/cows", summary="Get all registered cows")
def admin_get_all_cows(
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    cows = db.query(Cow).all()
    return {
        "total_cows": len(cows),
        "cows": [{
            "cow_id": c.cow_id,
            "cow_tag": c.cow_tag,
            "breed": c.breed,
            "color": c.color,
            "age": c.age,
            "owner_name": c.owner.full_name if c.owner else None,
            "owner_phone": c.owner.phone if c.owner else None,
            "registered_at": c.registration_date,
            "embeddings_count": len(c.embeddings),
            "qr_code_path": c.qr_code_path,
            "receipt_pdf_path": c.receipt_pdf_path,
            "qr_code_data": c.qr_code_data
        } for c in cows]
    }

# ---------------------------
# 5. Transfer Cow Ownership (Admin)
# ---------------------------
@router.put("/cows/{cow_id}/transfer", summary="Transfer cow ownership")
async def admin_transfer_ownership(
    background_tasks: BackgroundTasks,
    cow_id: int,
    # New Owner Details
    new_owner_full_name: str = Form(...),
    new_owner_email: Optional[str] = Form(None),
    new_owner_phone: Optional[str] = Form(None),
    new_owner_address: Optional[str] = Form(None),
    new_owner_national_id: Optional[str] = Form(None),
    # Optional cow updates
    breed: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    cow = db.query(Cow).filter(Cow.cow_id == cow_id).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    # Get current owner details
    old_owner = cow.owner
    
    # Create new owner
    new_owner = Owner(
        full_name=new_owner_full_name,
        email=new_owner_email,
        phone=new_owner_phone,
        address=new_owner_address,
        national_id=new_owner_national_id
    )
    db.add(new_owner)
    db.commit()
    db.refresh(new_owner)
    
    # Update cow ownership
    cow.previous_owner_id = cow.owner_id
    cow.owner_id = new_owner.owner_id
    cow.ownership_status = "transferred"
    cow.transfer_date = datetime.utcnow()
    
    # Update cow details if provided
    if breed: cow.breed = breed
    if color: cow.color = color
    if age: cow.age = age
    
    db.commit()
    
    # Generate transfer receipt
    if QR_AVAILABLE and new_owner.email:
        transfer_pdf_path = f"static/receipts/{cow.cow_id}_transfer_receipt.pdf"
        
        # Prepare data for transfer receipt
        cow_data = {
            "cow_id": cow.cow_id,
            "cow_tag": cow.cow_tag,
            "breed": cow.breed,
            "color": cow.color,
            "age": cow.age,
            "transfer_date": cow.transfer_date.strftime("%Y-%m-%d")
        }
        
        old_owner_data = {
            "full_name": old_owner.full_name,
            "email": old_owner.email,
            "phone": old_owner.phone
        }
        
        new_owner_data = {
            "full_name": new_owner.full_name,
            "email": new_owner.email,
            "phone": new_owner.phone,
            "address": new_owner.address,
            "national_id": new_owner.national_id
        }
        
        # Generate transfer receipt PDF
        generate_transfer_receipt_pdf(cow_data, old_owner_data, new_owner_data, cow.qr_code_path, transfer_pdf_path)
        
        # Update cow with transfer receipt path
        cow.transfer_receipt_path = transfer_pdf_path
        db.commit()
        
        # Send email to new owner
        background_tasks.add_task(main.send_transfer_email, new_owner.email, new_owner.full_name, old_owner.full_name, cow, transfer_pdf_path)
    
    return {
        "success": True,
        "message": "Ownership transferred successfully",
        "old_owner": old_owner.full_name,
        "new_owner": new_owner.full_name,
        "transfer_date": cow.transfer_date,
        "transfer_receipt_generated": bool(cow.transfer_receipt_path),
        "email_sent": bool(new_owner.email)
    }

# ---------------------------
# 6. Update Cow Details Only (Admin)
# ---------------------------
@router.put("/cows/{cow_id}", summary="Update cow details only")
def admin_update_cow_details(
    cow_id: int,
    breed: Optional[str] = Form(None),
    color: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    cow = db.query(Cow).filter(Cow.cow_id == cow_id).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    if breed: cow.breed = breed
    if color: cow.color = color
    if age: cow.age = age
    
    db.commit()
    return {"success": True, "message": "Cow details updated successfully"}

# ---------------------------
# 7. Delete Cow (Admin)
# ---------------------------
@router.delete("/cows/{cow_id}", summary="Delete cow")
def admin_delete_cow(
    cow_id: int,
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    cow = db.query(Cow).filter(Cow.cow_id == cow_id).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    db.delete(cow)
    db.commit()
    return {"success": True, "message": "Cow deleted successfully"}

# ---------------------------
# 10. Get Cow Tag Info
# ---------------------------
@router.get("/cow-tag/info", summary="Get cow tag format info")
def get_cow_tag_info(
    current_admin=Depends(get_current_admin)
):
    from datetime import datetime
    year = datetime.now().year
    return {
        "format": "TW-YYYY-OOO-XXXX",
        "description": {
            "TW": "Titweng prefix",
            "YYYY": f"Current year ({year})",
            "OOO": "Owner ID (3 digits)",
            "XXXX": "Random 4-character code"
        },
        "example": f"TW-{year}-001-A1B2",
        "auto_generated": True,
        "unique": True
    }

# ---------------------------
# 9. Verify Cow by Tag (Admin)
# ---------------------------
@router.post("/verify/tag", summary="Admin verify cow by tag")
def admin_verify_cow_by_tag(
    cow_tag: str = Form(...),
    location: Optional[str] = Form(None),
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    cow = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow tag not found")
    
    # Log verification
    log = VerificationLog(
        user_id=None,
        cow_id=cow.cow_id,
        verification_image="tag_verification",
        similarity_score=1.0,
        location=location or "Admin Dashboard",
        verified="yes",
        created_at=datetime.utcnow()
    )
    db.add(log)
    db.commit()
    
    # Check ownership status
    ownership_message = None
    if cow.ownership_status == "transferred":
        ownership_message = "⚠️ OWNERSHIP TRANSFERRED: This cow's ownership has been transferred. Please contact Titweng Team for full details."
    
    return {
        "verification_method": "cow_tag",
        "cow_tag": cow_tag,
        "verified": "yes",
        "similarity": 1.0,
        "ownership_status": cow.ownership_status,
        "ownership_message": ownership_message,
        "cow_details": {
            "cow_id": cow.cow_id,
            "cow_tag": cow.cow_tag,
            "breed": cow.breed,
            "color": cow.color,
            "owner_name": cow.owner.full_name if cow.owner else None,
            "transfer_date": cow.transfer_date.strftime('%Y-%m-%d') if cow.transfer_date else None
        }
    }

# ---------------------------
# 10. Verify Cow by Nose Print (Admin)
# ---------------------------
@router.post("/verify/nose", summary="Admin verify cow by nose print")
async def admin_verify_cow_by_nose(
    location: Optional[str] = Form(None),
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    if not files:
        raise HTTPException(status_code=400, detail="At least 1 image required")
    
    results = []
    for f in files[:2]:  # Max 2 images
        contents = await f.read()
            
        # Detect nose using HF API
        nose_result = main.detect_nose(contents)
        if nose_result is None:
            continue
        
        # Extract embedding using HF API
        query_emb = main.extract_embedding(contents)
        if query_emb is None:
            continue  # Skip this file if embedding extraction failed
        
        # Use pgvector for efficient similarity search
        from sqlalchemy import text
        
        # Convert embedding to string format for pgvector
        emb_str = '[' + ','.join(map(str, query_emb.tolist())) + ']'
        
        # Find most similar embedding using pgvector cosine similarity
        query = text("""
            SELECT cow_id, (1 - (embedding <=> :query_emb)) as similarity
            FROM embeddings 
            ORDER BY embedding <=> :query_emb 
            LIMIT 1
        """)
        
        result = db.execute(query, {"query_emb": emb_str}).fetchone()
        
        best_similarity = result.similarity if result else 0
        best_cow_id = result.cow_id if result else None
        
        # Require VERY high similarity for 100k+ scale (95%+)
        if best_similarity > 0.95:  # Ultra-strict threshold for large scale
            cow = db.query(Cow).filter(Cow.cow_id == best_cow_id).first()
            verified_status = "yes" if best_similarity > 0.97 else "partial"
            
            # Log verification
            log = VerificationLog(
                user_id=None,
                cow_id=best_cow_id,
                verification_image=f.filename,
                similarity_score=float(best_similarity),
                location=location or "Admin Dashboard",
                verified=verified_status,
                created_at=datetime.utcnow()
            )
            db.add(log)
            db.commit()
            
            result = {
                "file": f.filename,
                "verification_method": "nose_print",
                "cow_found": True,
                "similarity": float(round(best_similarity, 3)),
                "verified": verified_status,
                "cow_details": {
                    "cow_id": cow.cow_id,
                    "cow_tag": cow.cow_tag,
                    "breed": cow.breed,
                    "color": cow.color,
                    "owner_name": cow.owner.full_name if cow.owner else None
                }
            }
        else:
            result = {
                "file": f.filename,
                "verification_method": "nose_print",
                "cow_found": False,
                "similarity": float(round(best_similarity, 3)),
                "verified": "no",
                "message": "Cow not found in database"
            }
        
        results.append(result)
    
    return {"verification_results": results}

# ---------------------------
# 10. Get All Reports (Admin)
# ---------------------------
@router.get("/reports", summary="Get all reports")
def admin_get_reports(
    status: Optional[str] = None,
    report_type: Optional[str] = None,
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    query = db.query(Report)
    if status:
        query = query.filter(Report.status == status)
    if report_type:
        query = query.filter(Report.report_type == report_type)
    
    reports = query.order_by(Report.created_at.desc()).all()
    return {
        "total_reports": len(reports),
        "reports": [{
            "report_id": r.report_id,
            "reporter_name": r.reporter_name,
            "reporter_phone": r.reporter_phone,
            "cow_tag": r.cow_tag,
            "report_type": r.report_type,
            "subject": r.subject,
            "message": r.message,
            "location": r.location,
            "status": r.status,
            "admin_reply": r.admin_reply,
            "created_at": r.created_at
        } for r in reports]
    }

# ---------------------------
# 11. Reply to Report (Admin)
# ---------------------------
@router.put("/reports/{report_id}/reply", summary="Reply to report")
def admin_reply_report(
    report_id: int,
    admin_reply: str = Form(...),
    status: str = Form("resolved"),
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    report.admin_reply = admin_reply
    report.status = status
    report.updated_at = datetime.utcnow()
    
    db.commit()
    return {"success": True, "message": "Reply sent successfully"}

# ---------------------------
# 7. Get Verification Logs
# ---------------------------
@router.get("/verifications", summary="Get all verification logs")
def admin_get_verifications(
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    logs = db.query(VerificationLog).all()
    return {
        "total_verifications": len(logs),
        "verifications": [{
            "log_id": log.log_id,
            "cow_tag": log.cow.cow_tag if log.cow else None,
            "similarity_score": log.similarity_score,
            "verified": log.verified,
            "location": log.location,
            "created_at": log.created_at
        } for log in logs]
    }

# ---------------------------
# 8. Dashboard Stats (Admin)
# ---------------------------
@router.get("/dashboard/stats", summary="Get dashboard statistics")
def admin_dashboard_stats(
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    total_cows = db.query(Cow).count()
    total_owners = db.query(Owner).count()
    total_verifications = db.query(VerificationLog).count()
    successful_verifications = db.query(VerificationLog).filter(VerificationLog.verified == "yes").count()
    
    pending_reports = db.query(Report).filter(Report.status == "pending").count()
    
    return {
        "total_cows": total_cows,
        "total_owners": total_owners,
        "total_verifications": total_verifications,
        "successful_verifications": successful_verifications,
        "pending_reports": pending_reports
    }

# ---------------------------
# 9. Download Cow Receipt (Admin)
# ---------------------------
@router.get("/receipt/{cow_tag}", summary="Download cow receipt by tag")
def admin_download_receipt(
    cow_tag: str,
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    from fastapi.responses import FileResponse
    
    # Find cow by tag
    cow = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    # Check if receipt exists
    if not cow.receipt_pdf_path or not os.path.exists(cow.receipt_pdf_path):
        raise HTTPException(status_code=404, detail="Receipt not found")
    
    # Return file for download
    return FileResponse(
        path=cow.receipt_pdf_path,
        filename=f"Titweng_Receipt_{cow_tag}.pdf",
        media_type="application/pdf"
    )

@router.post("/receipt/info", summary="Get cow receipt info by tag")
def admin_get_receipt_info(
    cow_tag: str = Form(...),
    db: Session = Depends(get_db),
    current_admin=Depends(get_current_admin)
):
    # Find cow by tag
    cow = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow not found")
    
    return {
        "cow_tag": cow.cow_tag,
        "cow_id": cow.cow_id,
        "breed": cow.breed,
        "color": cow.color,
        "age": cow.age,
        "owner_name": cow.owner.full_name if cow.owner else None,
        "owner_email": cow.owner.email if cow.owner else None,
        "registration_date": cow.registration_date.strftime('%Y-%m-%d'),
        "receipt_available": bool(cow.receipt_pdf_path and os.path.exists(cow.receipt_pdf_path)),
        "receipt_path": cow.receipt_pdf_path,
        "qr_code_available": bool(cow.qr_code_path and os.path.exists(cow.qr_code_path)),
        "download_url": f"/admin/receipt/{cow_tag}" if cow.receipt_pdf_path else None
    }