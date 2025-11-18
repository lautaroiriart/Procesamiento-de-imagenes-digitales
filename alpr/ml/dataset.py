# alpr/ml/dataset.py
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from django.conf import settings

# Tamaño al que redimensionamos las imágenes de patentes
IMG_HEIGHT = 64
IMG_WIDTH = 256

# Rutas base de docs e imágenes
DOCS_DIR = Path(settings.BASE_DIR) / "docs"
IMAGES_DIR = DOCS_DIR / "images"
MAPPING_PATH = DOCS_DIR / "mapping_ocr_normal.xlsx"


def load_plate_dataset(
    mapping_path: Path = MAPPING_PATH,
    images_dir: Path = IMAGES_DIR,
    filename_col: str = "nuevo_nombre",
    plate_col: str = "patente",
):
    """
    Carga el dataset de patentes a partir del Excel de mapeo y las imágenes.

    mapping_ocr_normal.xlsx debe tener, al menos, las columnas:
        - nuevo_nombre : nombre del archivo de imagen (ej: '1.jpg')
        - PATENTE      : texto de la patente (ej: 'AE 451 UX')

    Devuelve:
        X: np.ndarray con shape (N, IMG_HEIGHT, IMG_WIDTH, 1)
        texts: lista de strings con las patentes.
    """

    df = pd.read_excel(mapping_path)

    if filename_col not in df.columns or plate_col not in df.columns:
        raise KeyError(
            f"No se encontraron las columnas requeridas '{filename_col}' y/o "
            f"'{plate_col}' en {mapping_path}. "
            f"Columnas disponibles: {list(df.columns)}"
        )

    images = []
    texts = []

    for _, row in df.iterrows():
        filename = str(row[filename_col]).strip()
        if not filename:
            continue

        # Patente: sacamos espacios y pasamos a mayúsculas
        plate_text = str(row[plate_col]).upper().replace(" ", "")
        if not plate_text:
            continue

        img_path = images_dir / filename
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

    if not images:
        raise RuntimeError("No se pudo cargar ninguna imagen para el dataset de OCR.")

    X = np.array(images)[..., np.newaxis]
    print(f"[INFO] Dataset cargado: {len(texts)} imágenes.")
    return X, texts
