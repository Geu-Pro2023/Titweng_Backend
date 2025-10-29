import random
import string
from datetime import datetime
from sqlalchemy.orm import Session
from models import Cow

def generate_secure_cow_tag(db: Session, owner_id: int) -> str:
    """Generate unique cow tag: TW-YYYY-OOO-XXXX"""
    year = datetime.now().year
    owner_str = f"{owner_id:03d}"  # 3-digit owner ID
    
    # Generate random 4-character code
    while True:
        random_code = ''.join(random.choices(string.ascii_uppercase + string.digits, k=4))
        cow_tag = f"TW-{year}-{owner_str}-{random_code}"
        
        # Check if tag already exists
        existing = db.query(Cow).filter(Cow.cow_tag == cow_tag).first()
        if not existing:
            return cow_tag