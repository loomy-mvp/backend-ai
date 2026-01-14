import smtplib
import os
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
import logging

load_dotenv(override=True)

logger = logging.getLogger(__name__)

SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
ERROR_RECEIVER_EMAIL = os.getenv("ERROR_RECEIVER_EMAIL", "loomy.mvp@gmail.com")


def send_error_email(subject: str, error_details: str, context: dict = None):
    """Send an email notification when an error occurs."""
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("[email] SMTP credentials not configured; skipping email notification")
        return

    try:
        msg = MIMEMultipart()
        msg["From"] = SMTP_USER
        msg["To"] = ERROR_RECEIVER_EMAIL
        msg["Subject"] = f"[Loomy Backend Error] {subject}"

        body = f"""
An error occurred in the Loomy backend:

Error: {error_details}

Context:
"""
        if context:
            for key, value in context.items():
                body += f"  - {key}: {value}\n"

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"[email] Error notification sent to {ERROR_RECEIVER_EMAIL}")
    except Exception as exc:
        logger.error(f"[email] Failed to send error notification: {exc}")
