import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

DATA_DIR = os.path.join("data", "raw")
os.makedirs(DATA_DIR, exist_ok=True)

api = KaggleApi()
api.authenticate()

DATASET = "yehongjiang/pokemon-sprites-images"
ZIP_NAME = "pokemon-sprites-images.zip"
ZIP_PATH = os.path.join(DATA_DIR, ZIP_NAME)

print(f"[INFO] Downloading {DATASET} ...")
api.dataset_download_files(DATASET, path=DATA_DIR, unzip=False, quiet=False)

if not os.path.exists(ZIP_PATH):
    raise FileNotFoundError(f"{ZIP_PATH} не найден после скачивания!")

print(f"[INFO] Unzipping dataset ...")
with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(DATA_DIR)

os.remove(ZIP_PATH)
print(f"[OK] Dataset downloaded and extracted to {DATA_DIR}")
