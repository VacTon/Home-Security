import smtplib
import ssl
from email.message import EmailMessage
import time
import logging
import threading

class Notifier:
    def __init__(self, config):
        self.config = config["email"]
        self.last_sent = {}  # Store last sent time for each label/person
        self.lock = threading.Lock()

    def send_email(self, subject, body, image=None):
        if not self.config["enabled"]:
            return

        msg = EmailMessage()
        msg.set_content(body)
        msg["Subject"] = subject
        msg["From"] = self.config["sender_email"]
        msg["To"] = ", ".join(self.config["receiver_emails"])

        context = ssl.create_default_context()

        try:
            logging.info(f"Sending email: {subject}")
            with smtplib.SMTP(self.config["smtp_server"], self.config["smtp_port"]) as server:
                server.starttls(context=context)
                server.login(self.config["sender_email"], self.config["sender_password"])
                server.send_message(msg)
            logging.info("Email sent successfully.")
        except Exception as e:
            logging.error(f"Failed to send email: {e}")

    def notify(self, label):
        """Sends a notification if outside the cooldown period."""
        now = time.time()
        cooldown = self.config.get("cooldown_seconds", 300)

        with self.lock:
            last_time = self.last_sent.get(label, 0)
            if now - last_time > cooldown:
                self.last_sent[label] = now
                
                if label == "Unknown":
                    subject = "Security Alert: Unknown Person Detected"
                    body = "An unknown person has been detected by your security camera."
                else:
                    subject = f"Welcome Home: {label}"
                    body = f"{label} has arrived home."

                # Run in a separate thread to not block the main loop
                threading.Thread(target=self.send_email, args=(subject, body)).start()
            else:
                logging.debug(f"Notification suppressed for {label} (Cooldown).")
