# alpr/ml/plate_utils.py

"""
Utilidades simples para validación y normalización de patentes.
Incluye:
- Corrección de caracteres ambiguos detectados por OCR.
- Validación mediante expresiones regulares.
"""

import re

# Patrones típicos de patentes argentinas (nuevas y viejas)
PLATE_PATTERNS = [
    # Formato MERCOSUR AA999ZZ, pero versión compacta usada en dataset:
    re.compile(r"^[A-Z]{3}[0-9][A-Z][0-9]{2}$"),

    # Formato más antiguo tipo ABC1234:
    re.compile(r"^[A-Z]{3}[0-9]{4}$"),
]

# Caracteres que el OCR suele confundir:
# O→0, I→1, B→8
AMBIGUITY_MAP = str.maketrans({
    "O": "0",
    "I": "1",
    "B": "8",
})


def fix_confusions(text: str) -> str:
    """
    Reemplaza caracteres frecuentemente confundidos por el OCR.
    Ejemplo: O→0, I→1, B→8.
    No modifica el largo de la cadena.
    """
    if not isinstance(text, str):
        return text

    clean_text = text.strip().upper()
    return clean_text.translate(AMBIGUITY_MAP)


def looks_like_plate(text: str) -> bool:
    """
    Determina si el texto coincide con alguno de los patrones de patente conocidos.
    """
    if not isinstance(text, str):
        return False

    candidate = text.strip().upper()
    return any(pattern.fullmatch(candidate) for pattern in PLATE_PATTERNS)
