# alpr/ml/inference.py
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model

<<<<<<< HEAD
from .encoding import decode_indices, MAX_LEN, ALPHABET
from .dataset import IMG_HEIGHT, IMG_WIDTH

_model_cache = None
=======
from .dataset import IMG_HEIGHT, IMG_WIDTH
from .encoding import ALPHABET, MAX_LEN
   
   
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
except ImportError:
    pytesseract = None

_model = None
>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)


def get_model():
    """
<<<<<<< HEAD
    Carga el modelo desde models/plate_ocr_model.h5 (una sola vez).
    """
    global _model_cache
    if _model_cache is None:
=======
    Carga el modelo plate_ocr_model.h5 una sola vez (singleton).
    """
    global _model
    if _model is None:
>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
        model_path = Path(settings.BASE_DIR) / "models" / "plate_ocr_model.h5"
        if not model_path.exists():
            raise FileNotFoundError(
                f"No se encontró el modelo entrenado en {model_path}. "
                f"Ejecutá primero: python manage.py train_ocr"
            )
<<<<<<< HEAD
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
=======
        _model = load_model(model_path)
    return _model


def _apply_brightness_contrast(img, brightness=0, contrast=0):
    """
    Aplica brillo y contraste simples con OpenCV.
    brightness: desplazamiento (-100 a 100 aprox)
    contrast: escala (-100 a 100 aprox)
    """
    beta = float(brightness)
    alpha = 1.0 + (contrast / 100.0)  # 0 -> 1.0, 100 -> 2.0, -100 -> 0.0
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return img


def _preprocess_image(image_path, brightness=0, contrast=0):
    """
    Carga la imagen desde disco, aplica brillo/contraste y la deja lista
    para el modelo: (1, H, W, 1)
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    img = _apply_brightness_contrast(img, brightness=brightness, contrast=contrast)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype("float32") / 255.0
    img = img[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)
    return img

 
def run_ocr(image_path, brightness=0, contrast=0):
    """
    Corre el OCR usando el modelo propio.

    Devuelve:
        {
          "plate_text": "ABC123",
          "confidence": 0.92
        }
    """


    model = get_model()
    x = _preprocess_image(image_path, brightness=brightness, contrast=contrast)

    # preds es una lista de largo MAX_LEN,
    # cada elemento con shape (1, num_classes)
    preds = model.predict(x, verbose=0)

    chars = []
    confs = []

    for i in range(MAX_LEN):
        logits = preds[i][0]          # (num_classes,)
        idx = int(np.argmax(logits))  # índice de la clase más probable
        conf = float(logits[idx])     # probabilidad de ese carácter

        # mapeamos índice -> carácter usando ALPHABET
        # (ALPHABET está definido en encoding.py)
        if 0 <= idx < len(ALPHABET):
            chars.append(ALPHABET[idx])
            confs.append(conf)

    plate_text = "".join(chars).strip()
    confidence = float(np.mean(confs)) if confs else 0.0
>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)

    return {
        "plate_text": plate_text,
        "confidence": confidence,
    }
<<<<<<< HEAD
=======

from .dataset import _clean_plate  # si no la tenés exportada, copiá la función acá

def run_external_ocr(image_path, brightness=0, contrast=0):
    """
    Corre un OCR 'clásico' (Tesseract) para comparar contra el modelo propio.
    Devuelve un dict con plate_text y 'confianza' aproximada.
    """
    if pytesseract is None:
        return {
            "plate_text": "",
            "confidence": 0.0,
            "error": "pytesseract no instalado en el entorno",
        }

    img = cv2.imread(str(image_path))
    if img is None:
        return {
            "plate_text": "",
            "confidence": 0.0,
            "error": "No se pudo leer la imagen",
        }

    # Aplicamos el mismo brillo/contraste
    img = _apply_brightness_contrast(img, brightness=brightness, contrast=contrast)

    # Convertimos a gris y binarizamos suave
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (IMG_WIDTH, IMG_HEIGHT))
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # Config de Tesseract: una sola línea, solo letras/números
    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    raw_text = pytesseract.image_to_string(gray, config=config)
    plate_text = _clean_plate(raw_text)

    # 'Confianza' casera: proporción de caracteres válidos (largo razonable)
    if len(plate_text) == 0:
        conf = 0.0
    else:
        # si se parece a largo 6-7, le damos más fe
        ideal_len = 7
        conf = min(1.0, len(plate_text) / ideal_len)

    return {
        "plate_text": plate_text,
        "confidence": conf,
    }
>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
