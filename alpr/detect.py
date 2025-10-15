import cv2
import numpy as np

def find_plate_bbox(img: np.ndarray):
    """
    Heurística simple: Canny -> dilatación -> contornos -> filtrar por razón de aspecto/área.
    Devuelve (x, y, w, h) o None.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    edges = cv2.Canny(gray, 80, 200)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    dil = cv2.dilate(edges, kernel, iterations=1)
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]
    best = None
    best_score = 0.0
    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        if area < 0.001 * w * h:  # descartar muy pequeños
            continue
        ratio = cw / (ch + 1e-6)
        # Placas MERCOSUR suelen tener ratio aprox 4:1 a 5:1
        if 2.5 <= ratio <= 6.5:
            roi = gray[y:y+ch, x:x+cw]
            score = area * (roi.std() + 1e-3)
            if score > best_score:
                best_score = score
                best = (x, y, cw, ch)
    return best
