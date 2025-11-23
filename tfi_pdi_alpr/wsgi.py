# tfi_pdi_alpr/wsgi.py

"""
Punto de entrada WSGI para el proyecto Django.
Se utiliza cuando la aplicación se despliega en servidores WSGI
(gunicorn, mod_wsgi, uWSGI, etc.).
"""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault(
    "DJANGO_SETTINGS_MODULE",
    "tfi_pdi_alpr.settings"
)

application = get_wsgi_application()
