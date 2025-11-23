# alpr/urls.py

"""
Rutas principales de la aplicación ALPR.
Incluye:
- Vista de carga de imágenes.
- Sub-routing hacia los endpoints de la API (namespace: alpr_api).
"""

from django.urls import path, include
from . import views

app_name = "alpr"

urlpatterns = [
    # Vista principal del módulo
    path("upload/", views.upload_view, name="upload"),

    # Rutas de la API del módulo ALPR
    path(
        "api/",
        include(("alpr.api_urls", "alpr_api"), namespace="alpr_api"),
    ),
]
