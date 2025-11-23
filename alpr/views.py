# alpr/views.py

from django.shortcuts import render


def upload_view(request):
    """
    Renderiza la interfaz principal para cargar y procesar imágenes.
    """
    return render(request, "alpr/upload.html")