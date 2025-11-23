import cv2
import numpy as np


def find_plate_bbox(image: np.ndarray):
    """
    Detecta el bounding box más probable de una patente dentro de una imagen RGB.

    Retorna:
        (x, y, width, height) si encuentra un candidato válido.
        None si no se detecta ningún bbox razonable.
    """

    # --- Preprocesamiento básico ---
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

    # --- Detección de bordes ---
    edges = cv2.Canny(gray, threshold1=80, threshold2=200)

    # --- Aumentar regiones conectadas ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    image_height, image_width = gray.shape[:2]
    image_area = image_width * image_height

    best_bbox = None
    best_score = 0.0

    # Rango típico de relación de aspecto (ancho/alto) de una patente MERCOSUR
    MIN_ASPECT_RATIO = 2.5
    MAX_ASPECT_RATIO = 6.5

    # Umbral mínimo de área para descartar ruido
    MIN_AREA_RATIO = 0.001  # 0.1% del área total

    # --- Evaluación de contornos ---
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = width * height

        # Filtrar contornos demasiado pequeños
        if area < MIN_AREA_RATIO * image_area:
            continue

        aspect_ratio = width / (height + 1e-6)
        if not (MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            continue

        # Se usa la variación de intensidad dentro del ROI como indicador de "estructura"
        roi = gray[y : y + height, x : x + width]
        texture_score = roi.std()  # borde + variación interna = más probable patente

        score = area * (texture_score + 1e-3)

        if score > best_score:
            best_score = score
            best_bbox = (x, y, width, height)

    return best_bbox
