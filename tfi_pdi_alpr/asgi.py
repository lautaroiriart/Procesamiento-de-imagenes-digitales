# tfi_pdi_alpr/asgi.py

"""
Punto de entrada ASGI para el proyecto Django.
Permite servir la aplicación usando servidores ASGI (Daphne, Uvicorn, etc.).
"""

import os
from django.core.asgi import get_asgi_application

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "tfi_pdi_alpr.settings"
)

application = get_asgi_application()
