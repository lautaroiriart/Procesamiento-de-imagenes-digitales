# alpr/ml/dataset.py

from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from django.conf import settings

# Dimensiones a las que se reescala cada imagen de patente
IMG_HEIGHT = 64
IMG_WIDTH = 256

# Rutas base del dataset
DOCS_DIR = Path(settings.BASE_DIR) / "docs"
IMAGES_DIR = DOCS_DIR / "images"
MAPPING_PATH = DOCS_DIR / "mapping_ocr_normal.xlsx"


def load_plate_dataset(
    mapping_path: Path = MAPPING_PATH,
    images_dir: Path = IMAGES_DIR,
    filename_column: str = "nuevo_nombre",
    plate_column: str = "patente",
):
    """
    Carga el dataset de patentes usando un archivo Excel que mapea
    nombres de imágenes con sus etiquetas (texto de patente).

    Retorna:
        images_array: np.ndarray con shape (N, IMG_HEIGHT, IMG_WIDTH, 1)
        plate_texts: lista de strings con cada patente en formato limpio.
    """

    mapping_df = pd.read_excel(mapping_path)

    if filename_column not in mapping_df.columns or plate_column not in mapping_df.columns:
        available = list(mapping_df.columns)
        raise KeyError(
            f"Columnas requeridas '{filename_column}' y/o '{plate_column}' no encontradas.\n"
            f"Columnas disponibles: {available}"
        )

    images = []
    plate_texts = []

    for _, row in mapping_df.iterrows():
        image_name = str(row[filename_column]).strip()
        if not image_name:
            continue

        text_raw = str(row[plate_column]).strip()
        if not text_raw:
            continue

        # Normalizar formato de patente
        plate_text = text_raw.upper().replace(" ", "")

        image_path = images_dir / image_name
        if not image_path.exists():
            print(f"[WARN] Imagen no encontrada: {image_path}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"[WARN] Fallo al leer imagen: {image_path}")
            continue

        # Preprocesamiento básico (tamaño + normalización)
        image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
        image = image.astype("float32") / 255.0

        images.append(image)
        plate_texts.append(plate_text)

    if not images:
        raise RuntimeError(
            "No se cargó ninguna imagen. Verificar estructura del dataset y rutas."
        )

    images_array = np.array(images)[..., np.newaxis]
    print(f"[INFO] Dataset cargado correctamente: {len(plate_texts)} imágenes.")

    return images_array, plate_texts
