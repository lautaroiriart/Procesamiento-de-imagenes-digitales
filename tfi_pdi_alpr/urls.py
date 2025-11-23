# tfi_pdi_alpr/urls.py

"""
Rutas principales del proyecto Django.
Incluye:
- Panel de administración
- Rutas de la aplicación ALPR
"""

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    # Administración de Django
    path("admin/", admin.site.urls),

    # Rutas de la aplicación ALPR
    path("alpr/", include("alpr.urls")),
]
