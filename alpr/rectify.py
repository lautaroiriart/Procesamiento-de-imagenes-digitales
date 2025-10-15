import numpy as np
import cv2

def warp_plate(img: np.ndarray, bbox):
    if bbox is None:
        return img
    x, y, w, h = bbox
    plate = img[y:y+h, x:x+w]
    target_w = 256
    target_h = 64
    plate = cv2.resize(plate, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    return plate
