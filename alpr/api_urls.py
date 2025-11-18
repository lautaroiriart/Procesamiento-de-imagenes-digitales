# alpr/api_urls.py
from django.urls import path
from . import api_views

<<<<<<< HEAD
=======

>>>>>>> 76b396c (Arreglo de logica y comparativa con Tesseract)
app_name = "alpr_api"

urlpatterns = [
    path("predict/", api_views.ocr_predict, name="predict"),
]

