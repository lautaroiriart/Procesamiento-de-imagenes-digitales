import io
import numpy as np
import cv2
from PIL import Image
from .detect import find_plate_bbox
from .rectify import warp_plate
from .segment import split_characters
from .ocr_model import load_model, predict_chars
from .postprocess import fix_confusions

_model = None

def _img_from_bytes(img_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

def get_model():
    global _model
    if _model is None:
        _model = load_model()  # pesos opcionales; arranca en modo 'stub'
    return _model

def infer_image(img_bytes: bytes):
    img = _img_from_bytes(img_bytes)
    bbox = find_plate_bbox(img)  # (x, y, w, h) o None
    plate = warp_plate(img, bbox) if bbox is not None else img
    char_imgs = split_characters(plate)
    model = get_model()
    preds, confs = predict_chars(model, char_imgs)
    text = fix_confusions("".join(preds)) if preds else None
    out = {
        "plate_text": text,
        "per_char_conf": confs if confs else None,
        "bbox": [int(b) for b in bbox] if bbox is not None else None,
    }
    return out
