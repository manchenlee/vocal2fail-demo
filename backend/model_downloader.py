# backend/model_downloader.py
import os
import subprocess
import zipfile

def ensure_model_exists():
    model_folder = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(model_folder):
        print("Downloading and extracting model...")
        # 下載 zip
        subprocess.run(["gdown", "--id", "1xH1p2vZ7EkN1mQGWDDNRw51ThdV29En2", "-O", "models.zip"])
        # 解壓縮
        with zipfile.ZipFile("models.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Model extracted.")
    else:
        print("Model already exists.")