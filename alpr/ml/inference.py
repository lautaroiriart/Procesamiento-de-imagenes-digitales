# alpr/ml/inference.py
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model

from .encoding import decode_indices, MAX_LEN, ALPHABET
from .dataset import IMG_HEIGHT, IMG_WIDTH

_model_cache = None


def get_model():
    """
    Carga el modelo desde models/plate_ocr_model.h5 (una sola vez).
    """
    global _model_cache
    if _model_cache is None:
        model_path = Path(settings.BASE_DIR) / "models" / "plate_ocr_model.h5"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado en {model_path}. "
                f"Ejecutá primero: python manage.py train_ocr"
            )
        _model_cache = load_model(model_path)
    return _model_cache


def preprocess_image(image_path, brightness=0, contrast=0):
    """
    Lee la imagen, aplica brillo/contraste, redimensiona y normaliza.
    brightness: -100..100 (aprox)
    contrast:   -100..100 (aprox)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None

    # contraste y brillo básicos
    alpha = 1.0 + (contrast / 100.0)  # factor de contraste
    beta = brightness                 # brillo
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))  # (1, H, W, 1)
    return img


def run_ocr(image_path, brightness=0, contrast=0):
    """
    Ejecuta el OCR de tu modelo:
    - devuelve plate_text SIN espacios
    - confidence: promedio de las confianzas por carácter (0–1)
    """
    model = get_model()
    x = preprocess_image(image_path, brightness=brightness, contrast=contrast)
    if x is None:
        return {"plate_text": None, "confidence": None}

    preds = model.predict(x)  # lista de MAX_LEN arrays (1, num_classes)
    preds = [p[0] for p in preds]  # sacamos dimensión de batch

    indices = [int(np.argmax(p)) for p in preds]
    confidences = [float(np.max(p)) for p in preds]

    plate_text = decode_indices(indices)
    confidence = float(np.mean(confidences)) if confidences else None

    return {
        "plate_text": plate_text,
        "confidence": confidence,
    }
