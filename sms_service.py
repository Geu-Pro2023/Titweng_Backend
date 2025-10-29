# sms_service.py - Twilio SMS notifications
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def send_verification_alert_sms(owner_phone: str, cow_tag: str, location: str = None):
    """Send SMS alert to owner when their cow is being verified"""
    try:
        from twilio.rest import Client
        
        # Twilio configuration
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        titweng_contact = os.getenv("TITWENG_CONTACT_NUMBER")
        
        if not all([account_sid, auth_token, twilio_phone]) or account_sid == "your_twilio_account_sid":
            print("❌ Twilio credentials not configured properly")
            return False
        
        client = Client(account_sid, auth_token)
        
        # Format phone number
        if not owner_phone.startswith('+'):
            owner_phone = f"+{owner_phone}"
        
        # Create message
        location_text = f" at {location}" if location else ""
        message_body = f"Hello, your cow {cow_tag} is being verified right now{location_text}. If this is you, ignore this message. If NOT you, call Titweng: +250792104851 for help."
        
        # Send SMS
        message = client.messages.create(
            body=message_body,
            from_=twilio_phone,
            to=owner_phone
        )
        
        print(f"✅ SMS sent to {owner_phone} for cow {cow_tag}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")
        return False

def send_suspicious_activity_sms(owner_phone: str, cow_tag: str, details: str):
    """Send SMS for suspicious verification activity"""
    try:
        from twilio.rest import Client
        
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")
        twilio_phone = os.getenv("TWILIO_PHONE_NUMBER")
        titweng_contact = os.getenv("TITWENG_CONTACT_NUMBER")
        
        if not all([account_sid, auth_token, twilio_phone]):
            return False
        
        client = Client(account_sid, auth_token)
        
        if not owner_phone.startswith('+'):
            owner_phone = f"+{owner_phone}"
        
        message_body = f"""
🚨 TITWENG SECURITY ALERT

Suspicious activity detected for your cow {cow_tag}

Details: {details}
Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

URGENT: Call Titweng immediately:
📞 {titweng_contact}

- Titweng Security Team
        """.strip()
        
        message = client.messages.create(
            body=message_body,
            from_=twilio_phone,
            to=owner_phone
        )
        
        print(f"🚨 Security alert SMS sent to {owner_phone}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to send security SMS: {e}")
        return False