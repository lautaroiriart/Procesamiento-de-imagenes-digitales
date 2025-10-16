
from django.shortcuts import render
from django.http import JsonResponse
from .ml.inference import infer

def upload_view(request):
    if request.method == "POST" and request.FILES.get("image"):
        img_bytes = request.FILES["image"].read()
        result = infer(img_bytes)
        if request.headers.get("x-requested-with") == "XMLHttpRequest":
            return JsonResponse(result)
        return render(request, "alpr/upload.html", {"result": result})
    return render(request, "alpr/upload.html")
