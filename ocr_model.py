import os, pathlib, numpy as np
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
try:
    import torch
    from alpr.ocr_net import SmallCNN, ALPHABET as _ALPHA2
    _TORCH_OK = True
except Exception:
    _TORCH_OK = False

def _torch_model(weights_path: str | None):
    model = SmallCNN(n_classes=len(ALPHABET))
    if weights_path and pathlib.Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    return model

def load_model(weights_path: str | None = None):
    if _TORCH_OK:
        default = weights_path or os.environ.get("OCR_WEIGHTS", "models/ocr_cnn.pt")
        if default and pathlib.Path(default).exists():
            try:
                return _torch_model(default)
            except Exception:
                pass
    return {"stub": True}

def _predict_chars_torch(model, char_imgs: list[np.ndarray]):
    xs = []
    for im in char_imgs:
        im = im.astype("float32") / 255.0
        if im.ndim == 3:
            im = im[...,0]
        im = im[None, None, :, :]
        xs.append(im)
    x = torch.tensor(np.stack(xs))
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)
    preds = [ALPHABET[i] for i in idx.tolist()]
    return preds, conf.tolist()

def predict_chars(model, char_imgs: list[np.ndarray]):
    if not char_imgs:
        return [], []
    if isinstance(model, dict) and model.get("stub"):
        preds = ["X"] * len(char_imgs)
        confs = [0.5] * len(char_imgs)
        return preds, confs
    return _predict_chars_torch(model, char_imgs)
