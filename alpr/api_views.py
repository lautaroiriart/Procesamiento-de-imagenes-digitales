# alpr/api_views.py
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile

from .ml.inference import run_ocr


@csrf_exempt
def ocr_predict(request):
    if request.method != "POST":
        return JsonResponse({"error": "Solo se permite POST"}, status=405)

    image_file = request.FILES.get("image")
    if not image_file:
        return JsonResponse({"error": "Falta el archivo de imagen"}, status=400)

    # sliders del front (por ahora opcionales)
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

    # tu modelo
    custom_result = run_ocr(full_path, brightness=brightness, contrast=contrast)

    # más adelante acá podemos sumar "external_ocr" con pytesseract/easyocr
    return JsonResponse({
        "custom_model": custom_result,
        # "external_ocr": external_result,
    })
