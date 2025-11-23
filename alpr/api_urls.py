# alpr/api_urls.py

"""
Rutas públicas de la API del módulo ALPR.
Incluye endpoint de predicción OCR.
"""

from django.urls import path
from . import api_views

app_name = "alpr_api"

urlpatterns = [
    path("predict/", api_views.ocr_predict, name="predict"),
]
