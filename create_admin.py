from database import SessionLocal
from models import User
import bcrypt

def create_admin():
    db = SessionLocal()
    
    # Check if admin already exists
    existing_admin = db.query(User).filter(User.username == "titweng").first()
    if existing_admin:
        print("Admin user already exists")
        db.close()
        return
    
    # Create admin user
    password = "titweng@2025"
    password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    admin = User(
        username="titweng",
        email="admin@titweng.com",
        password_hash=password_hash,
        role="admin"
    )
    
    db.add(admin)
    db.commit()
    print("✅ Admin user created successfully")
    print("Username: titweng")
    print("Password: titweng@2025")
    print("⚠️  Please change the password after first login!")
    
    db.close()

if __name__ == "__main__":
    create_admin()
