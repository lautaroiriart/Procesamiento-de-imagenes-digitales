# alpr/ml/character_segmentation.py

"""
Funciones de segmentación para separar caracteres individuales dentro de
una patente ya recortada.
"""

import cv2
import numpy as np


def _binarize(image: np.ndarray) -> np.ndarray:
    """
    Convierte la imagen RGB de la patente en un mapa binario (blanco y negro)
    robusto para segmentación de caracteres.

    - Convierte a escala de grises.
    - Aplica suavizado.
    - Usa umbral adaptativo invertido.
    - Usa Otsu como fallback si el resultado es demasiado claro u oscuro.
    - Aplica cierre morfológico para unir fragmentos de caracteres.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25,
        C=10,
    )

    # Si el binarizado quedó prácticamente vacío o completamente lleno,
    # usamos Otsu como fallback (más robusto).
    mean_value = binary.mean()
    if mean_value < 5 or mean_value > 250:
        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

    # Cierre morfológico para unir huecos entre trazos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)

    return binary


def split_characters(plate_image: np.ndarray) -> list[np.ndarray]:
    """
    Segmenta caracteres individuales dentro de una patente binarizada.

    Retorna una lista de imágenes 32x32, cada una conteniendo un carácter.

    - Binariza la imagen.
    - Encuentra contornos de caracteres.
    - Filtra por altura mínima y relación de aspecto.
    - Ordena de izquierda a derecha.
    - Recorta y reescala cada carácter a 32x32.
    """

    binary = _binarize(plate_image)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    height, width = binary.shape[:2]
    min_height = int(0.2 * height)  # al menos 20% del alto de la patente
    max_aspect_ratio = 1.6          # caracteres muy “acostados” se descartan
    min_width = 4                   # vigilar ruido mínimo

    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if h < min_height:
            continue

        if w < min_width:
            continue

        aspect_ratio = w / (h + 1e-6)
        if aspect_ratio > max_aspect_ratio:
            continue

        bounding_boxes.append((x, y, w, h))

    # Ordenar caracteres de izquierda a derecha
    bounding_boxes.sort(key=lambda box: box[0])

    characters = []

    for x, y, w, h in bounding_boxes:
        roi = binary[y:y + h, x:x + w]
        roi = cv2.resize(roi, (32, 32), interpolation=cv2.INTER_AREA)
        characters.append(roi)

    return characters
