
import os, pathlib, numpy as np
from django.conf import settings
from .ocr_net import SmallCNN, ALPHABET

try:
    import torch
    _TORCH = True
except Exception:
    _TORCH = False

def _torch_model(weights_path: str | None):
    m = SmallCNN(n_classes=len(ALPHABET))
    if weights_path and pathlib.Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        m.load_state_dict(state)
    m.eval()
    return m

def load_model(weights_path: str | None = None):
    if _TORCH:
        wp = weights_path or getattr(settings, "OCR_WEIGHTS", "models/ocr_cnn.pt")
        if wp and pathlib.Path(wp).exists():
            try: return _torch_model(wp)
            except Exception: pass
    return {"stub": True}

def _predict_torch(model, char_imgs):
    xs = []
    for im in char_imgs:
        im = im.astype("float32")/255.0
        if im.ndim == 3: im = im[...,0]
        xs.append(im[None, None, :, :])
    x = torch.tensor(np.stack(xs))
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)
        conf, idx = probs.max(dim=1)
    preds = [ALPHABET[i] for i in idx.tolist()]
    return preds, conf.tolist()

def predict_chars(model, char_imgs):
    if not char_imgs: return [], []
    if isinstance(model, dict) and model.get("stub"):
        return ["X"]*len(char_imgs), [0.5]*len(char_imgs)
    return _predict_torch(model, char_imgs)
