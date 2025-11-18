# tfi_pdi_alpr/urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("alpr/", include("alpr.urls")),  # todo lo de la app cuelga de /alpr/
]
