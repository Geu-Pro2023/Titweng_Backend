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
        nose_result = main.detect_nose(contents)
        if nose_result is None:
            continue
        
        # Extract embedding using HF API
        query_emb = main.extract_embedding(contents)
        if query_emb is None:
            continue  # Skip this file if embedding extraction failed
        
        # ROBUST VERIFICATION: Check against ALL embeddings for ALL cows
        from sqlalchemy import text
        
        # Convert embedding to string format for pgvector
        emb_str = '[' + ','.join(map(str, query_emb.tolist())) + ']'
        
        # Get ALL embeddings and their similarities - comprehensive search
        query = text("""
            SELECT e.cow_id, c.cow_tag, (1 - (e.embedding <=> :query_emb)) as similarity,
                   e.image_angle, e.quality_score, e.is_primary
            FROM embeddings e
            JOIN cows c ON e.cow_id = c.cow_id
            ORDER BY e.embedding <=> :query_emb
        """)
        
        all_results = db.execute(query, {"query_emb": emb_str}).fetchall()
        
        if not all_results:
            best_similarity = 0
            best_match = None
            print(f"❌ No embeddings found in database for comparison")
        else:
            # Find the absolute best match across ALL cows and ALL embeddings
            best_result = all_results[0]  # Already ordered by similarity
            best_similarity = best_result.similarity
            best_match = best_result.cow_id
            
            print(f"🔍 VERIFICATION ANALYSIS for {f.filename}:")
            print(f"   Total embeddings checked: {len(all_results)}")
            print(f"   Best match: Cow {best_result.cow_tag} (ID: {best_match})")
            print(f"   Best similarity: {best_similarity:.4f}")
            print(f"   Embedding type: {best_result.image_angle}")
            
            # Show top 3 matches for transparency
            print(f"   Top 3 matches:")
            for i, r in enumerate(all_results[:3]):
                print(f"     {i+1}. Cow {r.cow_tag}: {r.similarity:.4f} ({r.image_angle})")
        
        # ROBUST DECISION: Must exceed threshold for positive identification
        VERIFICATION_THRESHOLD = 0.85
        
        if best_similarity > VERIFICATION_THRESHOLD:
            # CHECK FOR AMBIGUOUS MATCHES (multiple cows with high similarity)
            high_matches = [r for r in all_results if r.similarity > VERIFICATION_THRESHOLD]
            
            if len(high_matches) > 1:
                # Check if there are multiple different cows above threshold
                unique_cows = set(r.cow_id for r in high_matches)
                if len(unique_cows) > 1:
                    print(f"⚠️ AMBIGUOUS MATCH DETECTED:")
                    for i, r in enumerate(high_matches[:3]):
                        print(f"   {i+1}. Cow {r.cow_tag}: {r.similarity:.4f}")
                    
                    # If top match is significantly better, proceed
                    if best_similarity - high_matches[1].similarity > 0.05:  # 5% gap
                        print(f"✅ Clear winner: {best_similarity:.4f} vs {high_matches[1].similarity:.4f}")
                    else:
                        print(f"❌ Too close to call: {best_similarity:.4f} vs {high_matches[1].similarity:.4f}")
                        result = {
                            "file": f.filename,
                            "verification_status": "AMBIGUOUS",
                            "similarity": float(round(best_similarity, 4)),
                            "cow_found": False,
                            "message": f"Multiple cows found with similar scores. Manual verification required.",
                            "ambiguous_matches": [
                                {"cow_tag": r.cow_tag, "similarity": round(r.similarity, 4)} 
                                for r in high_matches[:3] if r.cow_id in unique_cows
                            ]
                        }
                        results.append(result)
                        continue
            
            cow = db.query(Cow).filter(Cow.cow_id == best_match).first()
            
            # Send SMS alert to owner
            owner_notified = False
            if cow.owner and cow.owner.phone:
                owner_notified = send_verification_alert_sms(
                    owner_phone=cow.owner.phone,
                    cow_tag=cow.cow_tag,
                    location="Mobile App Verification"
                )
            
            print(f"✅ VERIFICATION SUCCESS: Cow {cow.cow_tag} identified with {best_similarity:.3f} confidence")
            
            result = {
                "file": f.filename,
                "verification_status": "VERIFIED",
                "similarity": float(round(best_similarity, 4)),
                "confidence_level": "HIGH" if best_similarity > 0.92 else "MEDIUM",
                "cow_found": True,
                "total_embeddings_checked": len(all_results) if 'all_results' in locals() else 0,
                "cow_details": {
                    "cow_id": cow.cow_id,
                    "cow_tag": cow.cow_tag,
                    "breed": cow.breed,
                    "color": cow.color,
                    "age": cow.age,
                    "owner_name": cow.owner.full_name if cow.owner else None,
                    "owner_phone": cow.owner.phone if cow.owner else None,
                    "facial_image_url": f"/{cow.facial_image_path}" if cow.facial_image_path else None
                },
                "owner_notified": owner_notified
            }
        else:
            print(f"❌ VERIFICATION FAILED: No match found (best: {best_similarity:.3f}, threshold: {VERIFICATION_THRESHOLD})")
            
            result = {
                "file": f.filename,
                "verification_status": "NOT_FOUND",
                "similarity": float(round(best_similarity, 4)),
                "confidence_level": "LOW",
                "cow_found": False,
                "total_embeddings_checked": len(all_results) if 'all_results' in locals() else 0,
                "threshold_used": VERIFICATION_THRESHOLD,
                "message": f"No cow found with similarity above {VERIFICATION_THRESHOLD} threshold"
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
    
    # Send SMS alert to owner
    owner_notified = False
    if cow.owner and cow.owner.phone:
        owner_notified = send_verification_alert_sms(
            owner_phone=cow.owner.phone,
            cow_tag=cow.cow_tag,
            location="Tag Scan Verification"
        )
    
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
# 6. Submit Report with GPS Location
# ---------------------------
@router.post("/report/gps", summary="Submit report with GPS coordinates")
def submit_report_with_gps(
    reporter_name: str = Form(...),
    reporter_phone: str = Form(...),
    reporter_email: str = Form(None),
    cow_tag: str = Form(None),
    report_type: str = Form(...),  # suspect, theft, other
    subject: str = Form(...),
    message: str = Form(...),
    latitude: float = Form(..., description="GPS latitude coordinate"),
    longitude: float = Form(..., description="GPS longitude coordinate"),
    db: Session = Depends(get_db)
):
    """Submit report with precise GPS coordinates from mobile app"""
    
    # Format GPS coordinates as location string
    gps_location = f"{latitude:.6f}° N, {longitude:.6f}° E"
    
    report = Report(
        reporter_name=reporter_name,
        reporter_phone=reporter_phone,
        reporter_email=reporter_email,
        cow_tag=cow_tag,
        report_type=report_type,
        subject=subject,
        message=message,
        location=gps_location
    )
    db.add(report)
    db.commit()
    db.refresh(report)
    
    return {
        "success": True,
        "report_id": report.report_id,
        "message": "Report with GPS location submitted successfully",
        "gps_coordinates": {
            "latitude": latitude,
            "longitude": longitude,
            "formatted_location": gps_location
        }
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
    nose_result = main.detect_nose(contents)
    if nose_result is None:
        return {
            "nose_detected": False,
            "message": "No nose detected in frame",
            "suggestion": "Position cow's nose in camera view"
        }
    
    detection_confidence = nose_result.get('confidence', 0.8)
    
    # Extract embedding using HF API
    query_emb = main.extract_embedding(contents)
    if query_emb is None:
        return {
            "nose_detected": True,
            "detection_confidence": round(detection_confidence, 3),
            "cow_found": False,
            "message": "Failed to extract embedding from image"
        }
    
    # Use pgvector for efficient similarity search
    from sqlalchemy import text
    
    # Convert embedding to string format for pgvector
    emb_str = '[' + ','.join(map(str, query_emb.tolist())) + ']'
    
    # Find most similar embedding using pgvector
    query = text("""
        SELECT cow_id, (1 - (embedding <=> :query_emb)) as similarity
        FROM embeddings 
        ORDER BY embedding <=> :query_emb 
        LIMIT 1
    """)
    
    result = db.execute(query, {"query_emb": emb_str}).fetchone()
    
    best_similarity = result.similarity if result else 0
    best_cow_id = result.cow_id if result else None
    
    # Adjusted threshold for practical use
    if best_similarity > 0.85:
        cow = db.query(Cow).filter(Cow.cow_id == best_cow_id).first()
        
        # Send SMS alert to owner
        owner_notified = False
        if cow.owner and cow.owner.phone:
            owner_notified = send_verification_alert_sms(
                owner_phone=cow.owner.phone,
                cow_tag=cow.cow_tag,
                location="Live Camera Verification"
            )
        
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
                "owner_name": cow.owner.full_name if cow.owner else None,
                "facial_image_url": f"/{cow.facial_image_path}" if cow.facial_image_path else None
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

# ---------------------------
# 7. Get Cow Face by Tag (Mobile)
# ---------------------------
@router.get("/cow/{cow_tag}/face", summary="Get cow facial image by tag")
def mobile_get_cow_face_by_tag(
    cow_tag: str,
    db: Session = Depends(get_db)
):
    """Mobile endpoint to get cow facial image and basic details by tag"""
    cow = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
    if not cow:
        raise HTTPException(status_code=404, detail="Cow tag not found")
    
    return {
        "cow_tag": cow.cow_tag,
        "facial_image_url": f"/{cow.facial_image_path}" if cow.facial_image_path else None,
        "breed": cow.breed,
        "color": cow.color,
        "age": cow.age,
        "owner_name": cow.owner.full_name if cow.owner else None,
        "registration_date": cow.registration_date.strftime('%Y-%m-%d'),
        "ownership_status": cow.ownership_status
    }