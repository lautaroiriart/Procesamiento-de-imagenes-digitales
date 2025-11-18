# alpr/urls.py
from django.urls import path, include
from . import views

app_name = "alpr"

urlpatterns = [
    path("upload/", views.upload_view, name="upload"),

    # Registramos la API con namespace "alpr_api"
    path(
        "api/",
        include(("alpr.api_urls", "alpr_api"), namespace="alpr_api"),
    ),
]
