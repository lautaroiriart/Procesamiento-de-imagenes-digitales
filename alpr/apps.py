# alpr/apps.py

from django.apps import AppConfig


class AlprConfig(AppConfig):
    """
    Configuración principal de la aplicación ALPR.
    """
    default_auto_field = "django.db.models.BigAutoField"
    name = "alpr"
