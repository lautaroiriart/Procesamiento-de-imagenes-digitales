# alpr/api_views.py
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

# Ruta explícita al ejecutable de Tesseract en Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


@csrf_exempt
def ocr_predict(request):
    """
    Endpoint que recibe una imagen + parámetros de brillo/contraste,
    ejecuta el modelo propio (CNN) y Tesseract, y devuelve un JSON
    con ambos resultados.
    """
    try:
        if request.method != "POST":
            return JsonResponse({"error": "Solo se permite POST"}, status=405)

        image_file = request.FILES.get("image")
        if not image_file:
            return JsonResponse({"error": "Falta el archivo de imagen"}, status=400)

        # sliders del front
        try:
            brightness = int(request.POST.get("brightness", 0))
            contrast = int(request.POST.get("contrast", 0))
        except ValueError:
            brightness = 0
            contrast = 0

        # guardamos temporalmente en media/
        saved_path = default_storage.save(
            image_file.name, ContentFile(image_file.read())
        )
        full_path = default_storage.path(saved_path)

        # ---------- 1) TU MODELO CNN ----------
        custom_result = run_ocr(full_path, brightness=brightness, contrast=contrast)

        # ---------- 2) TESSERACT ----------
        external_result = _run_tesseract(full_path, brightness, contrast)

        return JsonResponse(
            {
                "custom_model": custom_result,
                "external_ocr": external_result,
            }
        )

    except Exception as e:
        # Log en consola para que veas el traceback en runserver
        traceback.print_exc()
        return JsonResponse(
            {
                "error": "Error interno en el servidor",
                "detail": str(e),
            },
            status=500,
        )


def _run_tesseract(image_path: str, brightness: int = 0, contrast: int = 0):
    """
    Versión simple de Tesseract:
      - Lee la imagen en BGR
      - Aplica brillo/contraste
      - Convierte a escala de grises
      - Llama a Tesseract con psm 7 y whitelist
    Si no devuelve nada útil, intenta con psm 6.
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            return {
                "plate_text": None,
                "confidence": None,
                "error": "No se pudo leer la imagen para Tesseract",
            }

        # 1) Brillo / contraste (igual criterio que tu modelo)
        alpha = 1.0 + (contrast / 100.0)  # ganancia
        beta = brightness                  # bias
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

        # 2) Escala de grises
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        best_raw = ""

        # ===== INTENTO 1: psm 7 con whitelist =====
        config1 = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        raw1 = pytesseract.image_to_string(gray, config=config1)
        print("[TESSERACT RAW1]", repr(raw1))

        if raw1 and raw1.strip():
            best_raw = raw1

        # ===== INTENTO 2: psm 6 más laxo si lo anterior no tiró nada =====
        if not best_raw:
            config2 = "--psm 6"
            raw2 = pytesseract.image_to_string(gray, config=config2)
            print("[TESSERACT RAW2]", repr(raw2))
            if raw2 and raw2.strip():
                best_raw = raw2

        # Si después de los intentos no hay texto crudo útil:
        if not best_raw or not best_raw.strip():
            return {
                "plate_text": None,
                "confidence": None,
                "error": "Tesseract no detectó texto",
            }

        # -------- Limpieza: solo alfanuméricos + correcciones ----------
        clean = "".join(ch for ch in best_raw if ch.isalnum()).upper()
        clean = fix_confusions(clean)

        # Si al limpiar quedó vacío, usamos como fallback el crudo en mayúsculas
        if not clean:
            clean = best_raw.strip().upper()

        # Si aún así quedó vacío (solo whitespace), devolvemos error
        if not clean:
            return {
                "plate_text": None,
                "confidence": None,
                "error": "Tesseract no detectó texto (solo espacios)",
            }

        # Confianza aproximada, según si respeta un patrón de patente
        if looks_like_plate(clean):
            conf = 0.9
        else:
            conf = 0.5

        return {
            "plate_text": clean,
            "confidence": conf,
        }

    except Exception as e:
        traceback.print_exc()
        return {
            "plate_text": None,
            "confidence": None,
            "error": f"Error ejecutando Tesseract: {e}",
        }
