from __future__ import annotations

import os
import time
import imaplib
import email
import hashlib
from email.header import decode_header
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import smtplib
from pathlib import Path

from ddddocr import DdddOcr   


try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency
    load_dotenv = None

if load_dotenv:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=env_path)


class EmailCaptchaSolver:
    def __init__(self) -> None:
        self.captcha_email_count = 0
        self.body = os.getenv("CAPTCHA_EMAIL_BODY", "Please respond to this email with the captcha solution.")

        self.sender_email = os.getenv("CAPTCHA_EMAIL_SENDER", "")
        self.sender_password = os.getenv("CAPTCHA_EMAIL_PASSWORD", "")
        self.recipient_email = os.getenv("CAPTCHA_EMAIL_RECIPIENT", self.sender_email)

        self.smtp_server = os.getenv("CAPTCHA_EMAIL_SMTP_SERVER", "smtp.gmail.com")
        self.smtp_port = int(os.getenv("CAPTCHA_EMAIL_SMTP_PORT", "465"))
        self.imap_server = os.getenv("CAPTCHA_EMAIL_IMAP_SERVER", "imap.gmail.com")
        self._current_subject = None

    def is_configured(self) -> bool:
        return bool(self.sender_email and self.sender_password and self.recipient_email)

    def _captcha_subject(self) -> str:
        if self._current_subject:
            return self._current_subject
        return f"Captcha to Solve Trial Number {self.captcha_email_count}"

    def _send_email(
        self,
        subject: str,
        body: str,
        *,
        attachment_name: str | None = None,
        attachment_bytes: bytes | None = None,
    ) -> None:
        message = MIMEMultipart()
        message["Subject"] = subject
        message["From"] = self.sender_email
        message["To"] = self.recipient_email
        message.attach(MIMEText(body))
        if attachment_name and attachment_bytes is not None:
            message.attach(MIMEApplication(attachment_bytes, Name=attachment_name))

        with smtplib.SMTP_SSL(self.smtp_server, self.smtp_port) as server:
            server.login(self.sender_email, self.sender_password)
            server.sendmail(self.sender_email, self.recipient_email, message.as_string())

    def _send_email_with_captcha(self, img_bytes: bytes) -> None:
        self._send_email(
            subject=self._captcha_subject(),
            body=self.body,
            attachment_name="captcha.png",
            attachment_bytes=img_bytes,
        )

    def send_notification_email(self, subject: str, body: str) -> bool:
        if not self.is_configured():
            return False
        self._send_email(subject=subject, body=body)
        return True

    def _read_text_body(self, msg: email.message.Message) -> str:
        if msg.is_multipart():
            for part in msg.walk():
                if part.get_content_type() == "text/plain":
                    payload = part.get_payload(decode=True)
                    return payload.decode(errors="ignore") if payload else ""
            return ""
        payload = msg.get_payload(decode=True)
        return payload.decode(errors="ignore") if payload else ""

    def _get_email_with_captcha_solution(self, max_wait_s: int) -> str | None:
        start = time.time()
        while True:
            imap = imaplib.IMAP4_SSL(self.imap_server)
            imap.login(self.sender_email, self.sender_password)
            imap.select("inbox")

            _, messages = imap.search(None, "ALL")
            email_ids = messages[0].split()

            for email_id in list(reversed(email_ids))[:15]:
                _, msg_data = imap.fetch(email_id, "(RFC822)")
                raw_email = msg_data[0][1]
                msg = email.message_from_bytes(raw_email)

                subject, encoding = decode_header(msg.get("Subject", ""))[0]
                if isinstance(subject, bytes):
                    subject = subject.decode(encoding if encoding else "utf-8")

                if subject.startswith("Re:") and self._captcha_subject() in subject:
                    body = self._read_text_body(msg)
                    imap.logout()
                    self.captcha_email_count += 1
                    return body.strip()[:6]

            imap.logout()
            if max_wait_s > 0 and (time.time() - start) >= max_wait_s:
                return None
            time.sleep(5)

        return None

    def solve_captcha(self, img_bytes: bytes, max_wait_s: int) -> str | None:
        if not self.is_configured():
            return None
        token = hashlib.sha1(img_bytes).hexdigest()[:8]
        self._current_subject = f"Captcha to Solve {token} #{self.captcha_email_count}"
        self._send_email_with_captcha(img_bytes)
        result = self._get_email_with_captcha_solution(max_wait_s=max_wait_s)
        self._current_subject = None
        return result


def solve_captcha_api(img_bytes: bytes, api_key_env: str) -> str | None:
    ocr = DdddOcr(show_ad=False)
    captcha_solution = ocr.classification(img_bytes)
    print(captcha_solution)
    return "".join(c.upper() if c.isalpha() else c for c in captcha_solution)
