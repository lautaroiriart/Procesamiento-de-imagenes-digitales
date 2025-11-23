# alpr/ml/encoding.py

"""
Funciones de codificación y decodificación para patentes argentinas
formato MERCOSUR. Se utiliza codificación carácter-por-carácter.
"""

MAX_LEN = 7  # Longitud estándar AA123BB

# Conjunto permitido de caracteres
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# Mapas de codificación
char_to_index = {char: i for i, char in enumerate(ALPHABET)}
index_to_char = {i: char for i, char in enumerate(ALPHABET)}


def encode_text(text: str) -> list[int]:
    """
    Convierte una patente en una secuencia de índices de largo MAX_LEN.

    - Convierte el texto a mayúsculas.
    - Recorta si excede MAX_LEN.
    - Completa con '0' (carácter de padding) hasta MAX_LEN.
    - Caracteres no reconocidos → se codifican como índice 0.
    """
    if not isinstance(text, str):
        text = str(text)

    clean_text = text.upper().strip()

    # Ajustar longitud: recortar o rellenar con '0'
    clean_text = clean_text[:MAX_LEN].ljust(MAX_LEN, "0")

    encoded = [
        char_to_index.get(char, 0)  # fallback seguro
        for char in clean_text
    ]
    return encoded


def decode_indices(indices: list[int]) -> str:
    """
    Convierte una lista de índices a la patente correspondiente.

    - Usa index_to_char para reconstruir la secuencia.
    - Convierte cada índice a int para evitar errores de dtype.
    - Elimina '0' finales (padding).
    """
    chars = [
        index_to_char.get(int(idx), "")  # fallback a string vacío si algo no coincide
        for idx in indices
    ]

    plate = "".join(chars)

    # Remover padding final (ceros)
    return plate.rstrip("0")
