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

<<<<<<< HEAD
=======

>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
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

<<<<<<< HEAD
=======

>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
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

<<<<<<< HEAD
    filename_col = "nombre"   # nombre real de la columna en tu Excel
    plate_col = "patente"     # nombre real de la columna en tu Excel

=======
    # 🔹 NORMALIZAMOS NOMBRES DE COLUMNAS
    norm_cols = {str(c).strip().lower(): c for c in df.columns}

    # candidatos para la columna de filename
    filename_candidates = ["nombre", "nuevo_nombre", "origen"]
    filename_key = None
    for cand in filename_candidates:
        if cand in norm_cols:
            filename_key = cand
            break

    # candidatos para la columna de patente
    plate_candidates = ["patente"]
    plate_key = None
    for cand in plate_candidates:
        if cand in norm_cols:
            plate_key = cand
            break

    if filename_key is None or plate_key is None:
        raise ValueError(
            "No se pudieron identificar las columnas de nombre de archivo y patente.\n"
            f"Columnas disponibles: {list(df.columns)}"
        )

    filename_col = norm_cols[filename_key]   # p.ej. 'nuevo_nombre'
    plate_col = norm_cols[plate_key]         # p.ej. 'PATENTE'
>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
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
