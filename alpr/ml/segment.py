
import cv2, numpy as np

def _binarize(img):
    import cv2
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    bw = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    # Fallback Otsu si sale demasiado vacío
    if bw.mean() < 5 or bw.mean() > 250:
        _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    # Suavizar y cerrar huequitos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel, iterations=1)
    return bw

def split_characters(plate_img):
    import cv2, numpy as np
    bw = _binarize(plate_img)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bw.shape[:2]
    boxes = []
    for c in cnts:
        x,y,cw,ch = cv2.boundingRect(c)
        # MÁS permisivo:
        if ch < int(0.2*h) or cw < 4:
            continue
        if cw / (ch+1e-6) > 1.6:   # antes 1.2
            continue
        boxes.append((x,y,cw,ch))
    boxes.sort(key=lambda b: b[0])
    chars = []
    for (x,y,cw,ch) in boxes:
        roi = bw[y:y+ch, x:x+cw]
        roi = cv2.resize(roi, (32,32), interpolation=cv2.INTER_AREA)
        chars.append(roi)
    return chars

