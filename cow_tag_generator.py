# cow_tag_generator.py - Auto-generate unique cow tags
import random
import string
from datetime import datetime
from sqlalchemy.orm import Session

def generate_cow_tag(db: Session) -> str:
    """Generate unique cow tag: TW-YYYY-XXXXXX"""
    year = datetime.now().year
    
    while True:
        # Generate 6-character random alphanumeric code
        random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=6))
        cow_tag = f"TW-{year}-{random_code}"
        
        # Check if tag already exists
        from models import Cow
        existing = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
        if not existing:
            return cow_tag

def generate_secure_cow_tag(db: Session, owner_id: int) -> str:
    """Generate secure cow tag with owner reference: TW-YYYY-OWNERID-XXXX"""
    year = datetime.now().year
    
    while True:
        # Generate 4-character random code
        random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        cow_tag = f"TW-{year}-{owner_id:03d}-{random_code}"
        
        # Check if tag already exists
        from models import Cow
        existing = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
        if not existing:
            return cow_tag