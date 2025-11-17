# alpr/api_urls.py
from django.urls import path
from . import api_views

app_name = "alpr_api"

urlpatterns = [
    path("predict/", api_views.ocr_predict, name="predict"),
]

