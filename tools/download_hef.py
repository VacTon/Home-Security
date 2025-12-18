
import os
import requests
import logging

logging.basicConfig(level=logging.INFO)

# Base URL for Hailo Model Zoo (Hailo-8L / RPi5)
# Using v2.11.0 which is commonly used with RPi5 examples.
BASE_URL = "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.11.0/hailo8l"

# Models we want
MODELS = {
    "arcface_mobilefacenet.hef": "arcface_mobilefacenet.hef",
    # "arcface_r50.hef": "arcface_r50.hef", # Try mobilefacenet first, it's faster
    # "scrfd_2.5g.hef": "scrfd_2.5g.hef"   # Face detector (Optional)
}

OUTPUT_DIR = "models"

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        logging.info(f"File already exists: {dest_path}")
        return True
        
    logging.info(f"Downloading {url}...")
    try:
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(dest_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        logging.info(f"Downloaded to {dest_path}")
        return True
    except Exception as e:
        logging.error(f"Failed to download {url}: {e}")
        return False

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename, remote_name in MODELS.items():
        url = f"{BASE_URL}/{remote_name}"
        dest = os.path.join(OUTPUT_DIR, filename)
        download_file(url, dest)

if __name__ == "__main__":
    main()
