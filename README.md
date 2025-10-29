# Titweng Cattle Recognition API

FastAPI backend for cattle nose print recognition and management system.

## Features

- 🐄 Cattle registration with nose print biometrics
- 🔍 Mobile verification (no auth required)
- 👨‍💼 Admin dashboard (JWT protected)
- 📧 Email notifications with PDF receipts
- 📱 SMS alerts via Twilio
- 🤖 ML-powered nose detection and matching

## Tech Stack

- **Backend**: FastAPI, Python 3.9+
- **Database**: PostgreSQL with pgvector
- **ML**: Hugging Face APIs
- **Auth**: JWT tokens
- **Files**: PDF generation, QR codes

## Deployment

### Environment Variables (Set in Render)

```env
DATABASE_URL=postgresql://user:pass@host:port/db
ADMIN_USERNAME=titweng
ADMIN_PASSWORD=titweng@2025
SMTP_SERVER=smtp.gmail.com
SENDER_EMAIL=your-email@gmail.com
SENDER_PASSWORD=your-app-password
TWILIO_ACCOUNT_SID=your-twilio-sid
TWILIO_AUTH_TOKEN=your-twilio-token
TWILIO_PHONE_NUMBER=your-twilio-number
```

### Render Setup

1. Connect GitHub repo
2. Set environment variables
3. Deploy automatically

## API Endpoints

### Mobile (No Auth)
- `POST /mobile/verify/nose` - Verify cow by nose print
- `POST /mobile/verify/tag` - Verify cow by tag
- `POST /mobile/report` - Report suspicious activity

### Admin (JWT Required)
- `POST /admin/login` - Admin login
- `POST /admin/register-cow` - Register new cow
- `GET /admin/cows` - List all cows
- `PUT /admin/cows/{id}/transfer` - Transfer ownership
- `GET /admin/reports` - View reports

## Local Development

```bash
pip install -r requirements.txt
python create_admin.py
uvicorn main:app --reload
```
# Updated
