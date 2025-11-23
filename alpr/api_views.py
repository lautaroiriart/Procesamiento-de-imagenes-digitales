# alpr/api_views.py

"""
Endpoints de la API del módulo ALPR.
Incluye el endpoint de predicción OCR usando:
- Modelo propio (CNN Keras)
- OCR externo (Tesseract)
"""

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

import cv2
import numpy as np
import pytesseract
import traceback

from .ml.inference import run_ocr
from .ml.postprocess import fix_confusions, looks_like_plate


# Ruta del ejecutable de Tesseract en Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


@csrf_exempt
def ocr_predict(request):
    """
    Recibe una imagen + ajustes de brillo/contraste.
    Ejecuta:
        - El modelo CNN entrenado (custom_model).
        - OCR tradicional con Tesseract (external_ocr).

    Retorna un JSON con ambos resultados.
    """
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Solo se permite POST"}, status=405)

        image_file = request.FILES.get("image")
        if not image_file:
            return JsonResponse({"error": "Falta el archivo de imagen"}, status=400)

        brightness, contrast = _parse_adjustment_params(request)

        # Guardar archivo temporal en media/
        saved_path = default_storage.save(
            image_file.name,
            ContentFile(image_file.read())
        )
        full_path = default_storage.path(saved_path)

        # --- 1) Modelo propio ---
        custom_result = run_ocr(
            full_path,
            brightness=brightness,
            contrast=contrast
        )

        # --- 2) Tesseract ---
        external_result = _run_tesseract(
            full_path,
            brightness=brightness,
            contrast=contrast
        )

        return JsonResponse({
            "custom_model": custom_result,
            "external_ocr": external_result,
        })

    except Exception as e:
        traceback.print_exc()
        return JsonResponse(
            {"error": "Error interno en el servidor", "detail": str(e)},
            status=500
        )


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------

def _parse_adjustment_params(request):
    """
    Obtiene valores int de brillo y contraste.
    Si vienen inválidos, devuelve 0 como fallback.
    """
    try:
        brightness = int(request.POST.get("brightness", 0))
        contrast = int(request.POST.get("contrast", 0))
    except Exception:
        brightness = 0
        contrast = 0

    return brightness, contrast


def _run_tesseract(image_path: str, brightness: int = 0, contrast: int = 0):
    """
    Ejecuta OCR con Tesseract.

    Estrategia:
        1) Preprocesamiento básico (brillo/contraste, escala de grises).
        2) Tesseract con psm 7 y whitelist.
        3) Si no hay resultado, probar psm 6 (más laxo).
        4) Limpieza + correcciones de confusiones.
        5) Heurística básica de confianza.
    """

    try:
        img = cv2.imread(image_path)
        if img is None:
            return _tess_error("No se pudo leer la imagen para Tesseract")

        # --- Ajuste de brillo y contraste ---
        alpha = 1.0 + (contrast / 100.0)
        beta = brightness
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # ---- Intento 1: psm 7 + whitelist ----
        config_primary = (
            "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        )
        raw = pytesseract.image_to_string(gray, config=config_primary)
        print("[TESSERACT RAW1]", repr(raw))

        best_raw = raw.strip() if raw and raw.strip() else ""

        # ---- Intento 2: psm 6 como fallback ----
        if not best_raw:
            config_fallback = "--psm 6"
            raw2 = pytesseract.image_to_string(gray, config=config_fallback)
            print("[TESSERACT RAW2]", repr(raw2))

            if raw2 and raw2.strip():
                best_raw = raw2.strip()

        # ---- Si no hay nada útil ----
        if not best_raw:
            return _tess_error("Tesseract no detectó texto")

        # ---- Limpieza y corrección ----
        alnum = "".join(ch for ch in best_raw if ch.isalnum()).upper()
        cleaned = fix_confusions(alnum) if alnum else best_raw.upper()

        if not cleaned:
            return _tess_error("Tesseract no detectó texto (solo espacios)")

        # ---- Confianza heurística ----
        confidence = 0.9 if looks_like_plate(cleaned) else 0.5

        return {
            "plate_text": cleaned,
            "confidence": confidence,
        }

    except Exception as e:
        traceback.print_exc()
        return _tess_error(f"Error ejecutando Tesseract: {e}")


def _tess_error(message: str):
    """ Helper para devolver errores consistentes de Tesseract. """
    return {
        "plate_text": None,
        "confidence": None,
        "error": message,
    }
