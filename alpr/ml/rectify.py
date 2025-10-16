
import cv2, numpy as np
def warp_plate(img: np.ndarray, bbox):
    if bbox is None: return img
    x, y, w, h = bbox
    plate = img[y:y+h, x:x+w]
    return cv2.resize(plate, (256, 64), interpolation=cv2.INTER_CUBIC)
