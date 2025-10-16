
import cv2, numpy as np

def _binarize(img):
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    g = cv2.GaussianBlur(g, (3,3), 0)
    return cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 25, 10)

def split_characters(plate_img: np.ndarray):
    bw = _binarize(plate_img)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = bw.shape[:2]
    boxes = []
    for c in cnts:
        x,y,cw,ch = cv2.boundingRect(c)
        if ch < int(0.3*h) or cw < 6:
            continue
        if cw/ch > 1.2:
            continue
        boxes.append((x,y,cw,ch))
    boxes.sort(key=lambda b: b[0])
    chars = []
    for (x,y,cw,ch) in boxes:
        roi = bw[y:y+ch, x:x+cw]
        roi = cv2.resize(roi, (32,32), interpolation=cv2.INTER_AREA)
        chars.append(roi)
    return chars
