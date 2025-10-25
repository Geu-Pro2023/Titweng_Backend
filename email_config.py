import os

EMAIL_CONFIG = {
    "smtp_server": os.getenv("SMTP_SERVER", "smtp.gmail.com"),
    "smtp_port": int(os.getenv("SMTP_PORT", "587")),
    "sender_email": os.getenv("SENDER_EMAIL", "geu.bior@gmail.com"),
    "sender_password": os.getenv("SENDER_PASSWORD", ""),
    "sender_name": os.getenv("SENDER_NAME", "Titweng Cattle System")
}