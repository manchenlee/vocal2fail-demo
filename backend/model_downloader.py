# backend/model_downloader.py
import os
import gdown
import zipfile

def ensure_model_exists():
    model_folder = os.path.join(os.path.dirname(__file__), "models")
    if not os.path.exists(model_folder):
        print("Downloading and extracting model...")
        gdown.download('https://drive.google.com/uc?id=1xH1p2vZ7EkN1mQGWDDNRw51ThdV29En2', './models.zip', quiet=False)
        with zipfile.ZipFile("./models.zip", 'r') as zip_ref:
            zip_ref.extractall(".")
        print("Model extracted.")
    else:
        print("Model already exists.")