import requests
import time
import logging
import threading
import cv2
import os

class Notifier:
    def __init__(self, config):
        self.config = config["telegram"]
        self.bot_token = self.config["bot_token"]
        self.chat_id = self.config["chat_id"]
        self.last_sent = {}  # Store last sent time for each label/person
        self.lock = threading.Lock()
        
        # Test connection
        if self.config.get("enabled", True):
            self._test_connection()

    def _test_connection(self):
        """Test if bot token and chat ID are valid."""
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/getMe"
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                bot_info = response.json()
                logging.info(f"Telegram bot connected: @{bot_info['result']['username']}")
            else:
                logging.error(f"Telegram bot token invalid: {response.text}")
        except Exception as e:
            logging.error(f"Failed to connect to Telegram: {e}")

    def send_message(self, text, image_path=None):
        """Send a text message and optionally an image to Telegram."""
        if not self.config.get("enabled", True):
            return

        try:
            if image_path and os.path.exists(image_path):
                # Send photo with caption
                url = f"https://api.telegram.org/bot{self.bot_token}/sendPhoto"
                with open(image_path, 'rb') as photo:
                    files = {'photo': photo}
                    data = {'chat_id': self.chat_id, 'caption': text}
                    response = requests.post(url, files=files, data=data, timeout=10)
                    
                if response.status_code == 200:
                    logging.info(f"Telegram photo sent: {text[:50]}...")
                else:
                    logging.error(f"Failed to send Telegram photo: {response.text}")
            else:
                # Send text only
                url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
                data = {'chat_id': self.chat_id, 'text': text}
                response = requests.post(url, data=data, timeout=10)
                
                if response.status_code == 200:
                    logging.info(f"Telegram message sent: {text[:50]}...")
                else:
                    logging.error(f"Failed to send Telegram message: {response.text}")
                    
        except Exception as e:
            logging.error(f"Telegram send error: {e}")

    def notify(self, label, image_path=None):
        """Sends a notification if outside the cooldown period."""
        now = time.time()
        cooldown = self.config.get("cooldown_seconds", 300)

        with self.lock:
            last_time = self.last_sent.get(label, 0)
            if now - last_time > cooldown:
                self.last_sent[label] = now
                
                if label == "Unknown":
                    message = "🚨 *Security Alert*\n\nAn unknown person has been detected by your security camera."
                else:
                    message = f"🏠 *Welcome Home*\n\n{label} has arrived home."

                # Run in a separate thread to not block the main loop
                threading.Thread(
                    target=self.send_message, 
                    args=(message, image_path)
                ).start()
            else:
                logging.debug(f"Notification suppressed for {label} (Cooldown).")
