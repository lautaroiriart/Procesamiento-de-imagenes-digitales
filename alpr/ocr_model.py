import numpy as np

ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"

def load_model(weights_path: str | None = None):
    """Carga stub para demo inicial. Reemplazar por CNN entrenada desde cero."""
    return {"stub": True}

def predict_chars(model, char_imgs: list[np.ndarray]):
    if not char_imgs:
        return [], []
    preds = ["X"] * len(char_imgs)  # placeholder
    confs = [0.5] * len(char_imgs)
    return preds, confs
