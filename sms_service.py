# sms_service.py - Brevo SMS notifications
import os
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

def send_verification_alert_sms(owner_phone: str, cow_tag: str, location: str = None):
    """Send SMS alert to owner when their cow is being verified via Brevo API"""
    try:
        import requests
        
        # Brevo configuration
        brevo_api_key = os.getenv("BREVO_API_KEY")
        sms_sender = os.getenv("BREVO_SMS_SENDER", "Titweng")
        titweng_contact = os.getenv("TITWENG_CONTACT_NUMBER", "+250792104851")
        
        if not brevo_api_key:
            print("❌ Brevo API key not configured")
            return False
        
        # Format phone number
        if not owner_phone.startswith('+'):
            owner_phone = f"+{owner_phone}"
        
        # Professional message
        message_body = f"Titweng Security: Your registered cow {cow_tag} is currently being verified. If you did not request this verification, please contact our support team at {titweng_contact}."
        
        # Send SMS via Brevo API
        url = "https://api.brevo.com/v3/transactionalSMS/sms"
        headers = {
            "api-key": brevo_api_key,
            "Content-Type": "application/json"
        }
        
        data = {
            "sender": sms_sender,
            "recipient": owner_phone,
            "content": message_body
        }
        
        response = requests.post(url, json=data, headers=headers, timeout=30)
        
        if response.status_code == 201:
            print(f"✅ SMS sent via Brevo to {owner_phone} for cow {cow_tag}")
            return True
        else:
            print(f"❌ Brevo SMS API error: {response.status_code} - {response.text}")
            return False
        
    except Exception as e:
        print(f"❌ Failed to send SMS via Brevo: {e}")
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