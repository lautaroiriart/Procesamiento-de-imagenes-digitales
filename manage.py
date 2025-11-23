#!/usr/bin/env python
"""
Script de administración para el proyecto Django.
Permite ejecutar comandos como:
    python manage.py runserver
    python manage.py migrate
    python manage.py train_ocr
"""

import os
import sys


def main():
    """Punto de entrada principal para la ejecución de comandos Django."""
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "tfi_pdi_alpr.settings")

    from django.core.management import execute_from_command_line
    execute_from_command_line(sys.argv)


if __name__ == "__main__":
    main()
