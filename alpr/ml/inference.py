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
    Carga el modelo OCR una sola vez (lazy loading con caché).
    Retorna la instancia del modelo ya cargado.
    """
    global _model_cache

    if _model_cache is not None:
        return _model_cache

    model_path = Path(settings.BASE_DIR) / "models" / "plate_ocr_model.h5"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Modelo no encontrado en {model_path}. "
            "Debés entrenarlo primero con: python manage.py train_ocr"
        )

    _model_cache = load_model(model_path)
    return _model_cache


def preprocess_image(image_path, brightness=0, contrast=0):
    """
    Preprocesa la imagen para el modelo:

    - Lee en escala de grises.
    - Aplica corrección de contraste/brillo.
    - Redimensiona a tamaño estándar.
    - Normaliza a rango [0, 1].
    - Reordena a formato (1, H, W, 1) para el modelo.

    brightness: [-100, 100]     Cambia el valor de los píxeles.
    contrast:   [-100, 100]     Escala multiplicativa del contraste.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None

    # Ajuste básico de brillo y contraste
    alpha = 1.0 + (contrast / 100.0)   # factor multiplicativo
    beta = brightness                  # desplazamiento aditivo
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Redimensionar y normalizar
    image = cv2.resize(image, (IMG_WIDTH, IMG_HEIGHT))
    image = image.astype("float32") / 255.0

    # El modelo espera: (batch, alto, ancho, canales)
    image = np.expand_dims(image, axis=(0, -1))
    return image


def run_ocr(image_path, brightness=0, contrast=0):
    """
    Ejecuta el OCR usando el modelo entrenado.

    Retorna:
        - plate_text : texto predicho (sin espacios)
        - confidence : promedio de las confianzas por carácter (0–1)
    """
    model = get_model()
    preprocessed = preprocess_image(image_path, brightness=brightness, contrast=contrast)

    if preprocessed is None:
        return {"plate_text": None, "confidence": None}

    # Predicción del modelo: lista de MAX_LEN arrays (1, num_classes)
    raw_preds = model.predict(preprocessed)
    raw_preds = [p[0] for p in raw_preds]  # eliminar dimensión batch

    indices = [int(np.argmax(p)) for p in raw_preds]
    confidences = [float(np.max(p)) for p in raw_preds]

    # Decodificación estándar (quita padding '0')
    plate_text = decode_indices(indices)

    # Fallback si el modelo predice solo ceros
    if not plate_text:
        plate_text = "".join(ALPHABET[i] for i in indices)

    mean_confidence = float(np.mean(confidences)) if confidences else None

    return {
        "plate_text": plate_text,
        "confidence": mean_confidence,
    }
