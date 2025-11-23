import cv2
import numpy as np


def warp_plate(image: np.ndarray, bbox: tuple[int, int, int, int] | None) -> np.ndarray:
    """
    Extrae la región correspondiente a la patente utilizando un bounding box
    y la reescala al tamaño estándar (256x64).

    Parámetros:
        image : ndarray (H, W, C)
            Imagen original (RGB o escala de grises).
        bbox : (x, y, width, height) o None
            Bounding box de la patente.

    Retorna:
        ndarray : región recortada y reescalada, o la imagen original si no hay bbox.
    """
    if bbox is None:
        return image

    x, y, width, height = bbox

    # Validación defensiva (evita crasheos con bounding boxes fuera de rango)
    h, w = image.shape[:2]
    x_end = min(x + width, w)
    y_end = min(y + height, h)

    if x < 0 or y < 0 or x_end <= x or y_end <= y:
        # BBox inválido → retornamos imagen original
        return image

    plate_region = image[y:y_end, x:x_end]

    # Reescalado al tamaño estándar del OCR
    plate_resized = cv2.resize(
        plate_region,
        (256, 64),
        interpolation=cv2.INTER_CUBIC
    )

    return plate_resized
