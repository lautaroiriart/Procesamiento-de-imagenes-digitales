# alpr/api_views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
<<<<<<< HEAD
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .ml.inference import run_ocr
=======
from pathlib import Path
from django.conf import settings
import os

from .ml.inference import run_ocr, run_external_ocr
>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)


@csrf_exempt
def ocr_predict(request):
    if request.method != "POST":
<<<<<<< HEAD
        return JsonResponse({"error": "Solo se permite POST"}, status=405)

    image_file = request.FILES.get("image")
    if not image_file:
        return JsonResponse({"error": "Falta el archivo de imagen"}, status=400)

    # sliders del front (por ahora opcionales)
=======
        return JsonResponse({"error": "Método no permitido"}, status=405)

    image_file = request.FILES.get("image")
    if not image_file:
        return JsonResponse({"error": "No se envió imagen"}, status=400)

    # sliders
>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
    try:
        brightness = int(request.POST.get("brightness", 0))
        contrast = int(request.POST.get("contrast", 0))
    except ValueError:
        brightness = 0
        contrast = 0

<<<<<<< HEAD
    # guardamos temporalmente en media/
    saved_path = default_storage.save(
        image_file.name, ContentFile(image_file.read())
    )
    full_path = default_storage.path(saved_path)

    # tu modelo
    custom_result = run_ocr(full_path, brightness=brightness, contrast=contrast)

    # más adelante acá podemos sumar "external_ocr" con pytesseract/easyocr
    return JsonResponse({
        "custom_model": custom_result,
        # "external_ocr": external_result,
    })
=======
    # Guardamos imagen temporalmente en media/tmp
    tmp_dir = Path(settings.MEDIA_ROOT) / "tmp"
    tmp_dir.mkdir(exist_ok=True)
    tmp_path = tmp_dir / image_file.name

    with open(tmp_path, "wb") as f:
        for chunk in image_file.chunks():
            f.write(chunk)

    # Modelo propio (CNN)
    custom_result = run_ocr(tmp_path, brightness=brightness, contrast=contrast)

    # OCR clásico (Tesseract)
    external_result = run_external_ocr(tmp_path, brightness=brightness, contrast=contrast)

    # Borramos el archivo temporal
    try:
        os.remove(tmp_path)
    except OSError:
        pass

    return JsonResponse(
        {
            "custom_model": custom_result,
            "external_ocr": external_result,
        }
    )
>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
