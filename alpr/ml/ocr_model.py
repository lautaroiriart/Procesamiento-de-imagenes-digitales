# alpr/ml/pytorch_backend.py

import pathlib
import numpy as np
from django.conf import settings
from .ocr_net import SmallCNN, ALPHABET

try:
    import torch
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False


def load_torch_model(weights_path: str | None):
    """
    Inicializa SmallCNN y carga pesos si están disponibles.
    """
    model = SmallCNN(n_classes=len(ALPHABET))

    if weights_path and pathlib.Path(weights_path).exists():
        state = torch.load(weights_path, map_location="cpu")
        model.load_state_dict(state)

    model.eval()
    return model


def load_model(weights_path: str | None = None):
    """
    Carga el modelo OCR si PyTorch está disponible.

    Si no hay PyTorch o los pesos no existen:
        retorna un diccionario 'stub' para fallback controlado.
    """
    if TORCH_AVAILABLE:
        selected_path = (
            weights_path
            or getattr(settings, "OCR_WEIGHTS", "models/ocr_cnn.pt")
        )

        if selected_path and pathlib.Path(selected_path).exists():
            try:
                return load_torch_model(selected_path)
            except Exception:
                pass  # fallback abajo

    return {"stub": True}


def _prepare_char_tensor(char_images):
    """
    Normaliza imágenes de caracteres y crea un batch tensor.
    """
    processed = []

    for image in char_images:
        image = image.astype("float32") / 255.0

        # Convertir HxWxC → HxW si llega en grayscale con canal redundante
        if image.ndim == 3:
            image = image[..., 0]

        # PyTorch espera (batch, channels, height, width)
        processed.append(image[None, None, :, :])

    batch = np.stack(processed)
    return torch.tensor(batch)


def _predict_with_torch(model, char_images):
    """
    Ejecuta el modelo carácter-por-carácter.
    Retorna lista de caracteres predichos y sus confianzas.
    """
    batch = _prepare_char_tensor(char_images)

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1)

    confidences, indices = probs.max(dim=1)

    predicted_chars = [ALPHABET[i] for i in indices.tolist()]
    confidence_scores = confidences.tolist()

    return predicted_chars, confidence_scores


def predict_chars(model, char_images):
    """
    Predice caracteres usando el modelo PyTorch o un stub.

    Si recibe un stub:
        devuelve caracteres 'X' y confianza fija 0.5 para cada entrada.
    """
    if not char_images:
        return [], []

    if isinstance(model, dict) and model.get("stub"):
        return ["X"] * len(char_images), [0.5] * len(char_images)

    return _predict_with_torch(model, char_images)
