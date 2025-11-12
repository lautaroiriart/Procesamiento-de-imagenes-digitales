
import io, numpy as np
from PIL import Image
from .detect import find_plate_bbox
from .rectify import warp_plate
from .segment import split_characters
from .ocr_model import load_model, predict_chars
from .postprocess import fix_confusions

_model = None
def _as_rgb(img_bytes: bytes) -> np.ndarray:
    return np.array(Image.open(io.BytesIO(img_bytes)).convert("RGB"))

def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model

def infer(img_bytes: bytes):
    img = _as_rgb(img_bytes)
    bbox = find_plate_bbox(img)
    plate = warp_plate(img, bbox) if bbox is not None else img
    chars = split_characters(plate)
    model = get_model()
    preds, confs = predict_chars(model, chars)
    text = fix_confusions("".join(preds)) if preds else None
    return {"plate_text": text, "per_char_conf": confs or None,
            "bbox": list(map(int, bbox)) if bbox else None}

# al comienzo del archivo
from pathlib import Path, PurePath
DEBUG_DIR = Path("media/debug"); DEBUG_DIR.mkdir(parents=True, exist_ok=True)

# dentro de infer(), después de warp_plate:
from PIL import Image
Image.fromarray(plate).save(DEBUG_DIR / "plate_rectified.png")

# dentro de split_characters (al final), guardá el BW:
cv2.imwrite(str(DEBUG_DIR / "plate_bw.png"), bw)
