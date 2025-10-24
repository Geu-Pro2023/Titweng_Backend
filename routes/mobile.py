# routes/mobile.py - Mobile App Endpoints
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from sqlalchemy.orm import Session
from typing import List
import numpy as np
# Removed torch dependency

import main
from database import get_db
from models import Cow, Embedding, Report
# Remove preprocess_image import as it's no longer needed
from sms_service import send_verification_alert_sms

router = APIRouter(
    prefix="/mobile",
    tags=["Mobile App"]
)

# ---------------------------
# 1. Verify Cow by Nose Print
# ---------------------------
# ---------------------------
# 1. Verify Cow by Nose Print
# ---------------------------
@router.post("/verify/nose", summary="Verify cow by nose print")
async def verify_cow_by_nose(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db)
):
    if not files:
        raise HTTPException(status_code=400, detail="At least 1 image required")
    
    results = []
    for f in files[:2]:  # Max 2 images for mobile
        contents = await f.read()
        
        # Detect nose using HF API
        nose = main.detect_nose(contents)
        if nose is None:
            continue
        
        # Extract embedding using HF API
        query_emb = main.extract_embedding(nose)
        
        # Find best match across all cows
        all_embeddings = db.query(Embedding).all()
        best_match = None
        best_similarity = 0
        
        for emb_record in all_embeddings:
            stored_emb = np.array(emb_record.embedding)
            cos_sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
            
            if cos_sim > best_similarity:
                best_similarity = cos_sim
                best_match = emb_record.cow_id
        
        # Get cow details if match found
        if best_similarity > 0.75:
            cow = db.query(Cow).filter(Cow.cow_id == best_match).first()
            
            # Send SMS alert to owner (disabled until Twilio configured)
            # if cow.owner and cow.owner.phone:
            #     send_verification_alert_sms(
            #         owner_phone=cow.owner.phone,
            #         cow_tag=cow.cow_tag,
            #         location="Mobile App Verification"
            #     )
            
            result = {
                "file": f.filename,
                "similarity": float(round(best_similarity, 3)),
                "cow_found": True,
                "cow_details": {
                    "cow_id": cow.cow_id,
                    "cow_tag": cow.cow_tag,
                    "breed": cow.breed,
                    "color": cow.color,
                    "owner_name": cow.owner.full_name if cow.owner else None
                },
                "owner_notified": bool(cow.owner and cow.owner.phone)
            }
        else:
            result = {
                "file": f.filename,
                "similarity": float(round(best_similarity, 3)),
                "cow_found": False,
                "message": "Cow not found in database"
            }
        
        results.append(result)
    
    return {"verification_results": results}

# ---------------------------
# 2. Verify Cow by Tag
# ---------------------------
@router.get("/verify/tag/{cow_tag}", summary="Verify cow by tag")
def verify_cow_by_tag(
    cow_tag: str,
    db: Session = Depends(get_db)
):
    cow = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
    if not cow:
        return {"cow_found": False, "message": "Cow tag not found"}
    
    # Check ownership status
    ownership_message = None
    if cow.ownership_status == "transferred":
        ownership_message = "⚠️ OWNERSHIP TRANSFERRED: This cow's ownership has been transferred. Please contact Titweng Team for full details."
    
    # Send SMS alert to owner (disabled until Twilio configured)
    owner_notified = False
    # if cow.owner and cow.owner.phone:
    #     owner_notified = send_verification_alert_sms(
    #         owner_phone=cow.owner.phone,
    #         cow_tag=cow.cow_tag,
    #         location="Tag Scan Verification"
    #     )
    
    return {
        "cow_found": True,
        "owner_notified": owner_notified,
        "ownership_status": cow.ownership_status,
        "ownership_message": ownership_message,
        "cow_details": {
            "cow_id": cow.cow_id,
            "cow_tag": cow.cow_tag,
            "breed": cow.breed,
            "color": cow.color,
            "age": cow.age,
            "owner_name": cow.owner.full_name if cow.owner else None,
            "owner_phone": cow.owner.phone if cow.owner else None,
            "registered_date": cow.registration_date,
            "transfer_date": cow.transfer_date.strftime('%Y-%m-%d') if cow.transfer_date else None
        }
    }

# ---------------------------
# 3. Submit Report
# ---------------------------
@router.post("/report", summary="Submit report")
def submit_report(
    reporter_name: str = Form(...),
    reporter_phone: str = Form(...),
    reporter_email: str = Form(None),
    cow_tag: str = Form(None),
    report_type: str = Form(...),  # suspect, theft, other
    subject: str = Form(...),
    message: str = Form(...),
    location: str = Form(None),
    db: Session = Depends(get_db)
):
    report = Report(
        reporter_name=reporter_name,
        reporter_phone=reporter_phone,
        reporter_email=reporter_email,
        cow_tag=cow_tag,
        report_type=report_type,
        subject=subject,
        message=message,
        location=location
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    
    return {
        "success": True,
        "report_id": report.report_id,
        "message": "Report submitted successfully"
    }

# ---------------------------
# 4. Get Report Status
# ---------------------------
@router.get("/report/{report_id}", summary="Get report status")
def get_report_status(
    report_id: int,
    db: Session = Depends(get_db)
):
    report = db.query(Report).filter(Report.report_id == report_id).first()
    if not report:
        raise HTTPException(status_code=404, detail="Report not found")
    
    return {
        "report_id": report.report_id,
        "subject": report.subject,
        "status": report.status,
        "admin_reply": report.admin_reply,
        "created_at": report.created_at,
        "updated_at": report.updated_at
    }

# ---------------------------
# 5. Live Camera Verification
# ---------------------------
@router.post("/verify/live", summary="Live camera verification")
async def verify_cow_live_camera(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """Real-time verification from live camera feed"""
    contents = await file.read()
        
    # Detect nose using HF API
    nose = main.detect_nose(contents)
    if nose is None:
        return {
            "nose_detected": False,
            "message": "No nose detected in frame",
            "suggestion": "Position cow's nose in camera view"
        }
    
    detection_confidence = 0.8
    
    # Extract embedding using HF API
    query_emb = main.extract_embedding(nose)
    
    # Find best match across all cows
    all_embeddings = db.query(Embedding).all()
    best_match = None
    best_similarity = 0
    best_cow_id = None
    
    for emb_record in all_embeddings:
        stored_emb = np.array(emb_record.embedding)
        cos_sim = np.dot(query_emb, stored_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(stored_emb))
        
        if cos_sim > best_similarity:
            best_similarity = cos_sim
            best_cow_id = emb_record.cow_id
    
    # Get cow details if match found
    if best_similarity > 0.75:
        cow = db.query(Cow).filter(Cow.cow_id == best_cow_id).first()
        
        # Send SMS alert to owner (disabled until Twilio configured)
        owner_notified = False
        # if cow.owner and cow.owner.phone:
        #     owner_notified = send_verification_alert_sms(
        #         owner_phone=cow.owner.phone,
        #         cow_tag=cow.cow_tag,
        #         location="Live Camera Verification"
        #     )
        
        return {
            "nose_detected": True,
            "detection_confidence": round(detection_confidence, 3),
            "cow_found": True,
            "similarity": float(round(best_similarity, 3)),
            "verification_status": "verified" if best_similarity > 0.85 else "partial_match",
            "owner_notified": owner_notified,
            "cow_details": {
                "cow_id": cow.cow_id,
                "cow_tag": cow.cow_tag,
                "breed": cow.breed,
                "color": cow.color,
                "owner_name": cow.owner.full_name if cow.owner else None
            }
        }
    else:
        return {
            "nose_detected": True,
            "detection_confidence": round(detection_confidence, 3),
            "cow_found": False,
            "similarity": float(round(best_similarity, 3)),
            "verification_status": "not_found",
            "message": "Cow not found in database"
        }