# tfi_pdi_alpr/settings.py

"""
Configuración principal del proyecto Django para el TFI de Procesamiento Digital de Imágenes.
Incluye:
- Ajustes base de Django
- Configuración de plantillas
- Paths estáticos y media
- Ajustes para el módulo ALPR y sus modelos OCR
"""

from pathlib import Path
import os

# ---------------------------------------------------------
# Paths base
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------
# Seguridad / modo de ejecución
# ---------------------------------------------------------

SECRET_KEY = "dev-key"       # En producción, reemplazar por variable de entorno
DEBUG = True                 # En producción, usar False
ALLOWED_HOSTS = ["*"]        # Permite desarrollo desde cualquier host


# ---------------------------------------------------------
# Aplicaciones instaladas
# ---------------------------------------------------------

INSTALLED_APPS = [
    # Django apps nativas
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",

    # Librerías externas
    "rest_framework",

    # Apps del proyecto
    "alpr",
]


# ---------------------------------------------------------
# Middlewares
# ---------------------------------------------------------

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]


# ---------------------------------------------------------
# Root URLs y aplicaciones WSGI/ASGI
# ---------------------------------------------------------

ROOT_URLCONF = "tfi_pdi_alpr.urls"

WSGI_APPLICATION = "tfi_pdi_alpr.wsgi.application"
ASGI_APPLICATION = "tfi_pdi_alpr.asgi.application"


# ---------------------------------------------------------
# Templates
# ---------------------------------------------------------

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],  # Templates globales
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    }
]


# ---------------------------------------------------------
# Base de datos
# ---------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}


# ---------------------------------------------------------
# Internacionalización
# ---------------------------------------------------------

LANGUAGE_CODE = "es"
TIME_ZONE = "America/Argentina/Buenos_Aires"
USE_I18N = True
USE_TZ = True


# ---------------------------------------------------------
# Archivos estáticos y media
# ---------------------------------------------------------

STATIC_URL = "/static/"
STATICFILES_DIRS = [BASE_DIR / "static"]

MEDIA_URL = "/media/"
MEDIA_ROOT = BASE_DIR / "media"


# ---------------------------------------------------------
# Configuración ALPR / OCR
# ---------------------------------------------------------

# Pesos del modelo OCR basado en PyTorch
OCR_WEIGHTS = str(BASE_DIR / "models" / "ocr_cnn.pt")
