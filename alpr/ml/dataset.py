# alpr/ml/dataset.py
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from django.conf import settings

IMG_HEIGHT = 64
IMG_WIDTH = 256

DOCS_DIR = Path(settings.BASE_DIR) / "docs"
IMAGES_DIR = DOCS_DIR / "images"
MAPPING_PATH = DOCS_DIR / "mapping_ocr_normal.xlsx"

def _clean_plate(text: str) -> str:
    """
    Limpia la patente:
    - pasa a mayúsculas
    - elimina espacios
    - deja solo letras y números
    """
    if not isinstance(text, str):
        text = str(text)

    text = text.upper()
    # sacamos espacios
    text = text.replace(" ", "")
    # nos quedamos solo con A-Z y 0-9
    text = "".join(ch for ch in text if ch.isalnum())
    return text

def load_plate_dataset():
    """
    Lee mapping_ocr_normal.xlsx y carga las imágenes desde docs/images.
    Devuelve:
        X: array (N, H, W, 1)
        y_texts: lista de patentes limpias, sin espacios (N)
    """
    if not MAPPING_PATH.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {MAPPING_PATH}")

    df = pd.read_excel(MAPPING_PATH)

    filename_col = "nombre"   # nombre real de la columna en tu Excel
    plate_col = "patente"     # nombre real de la columna en tu Excel

    images = []
    texts = []

    for _, row in df.iterrows():
        filename = str(row[filename_col]).strip()
        plate_text = _clean_plate(row[plate_col])

        img_path = IMAGES_DIR / filename

        if not img_path.exists():
            print(f"[WARN] Imagen no encontrada: {img_path}")
            continue

        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Error al leer imagen: {img_path}")
            continue

        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = img.astype("float32") / 255.0

        images.append(img)
        texts.append(plate_text)

    X = np.array(images)[..., np.newaxis]
    print(f"[INFO] Dataset cargado: {len(texts)} imágenes.")
    return X, texts
