# alpr/ml/ocr_net.py

"""
Red neuronal CNN sencilla para clasificación de caracteres individuales.
Pensada para OCR carácter-por-carácter (licencia MERCOSUR).
"""

# Intentar importar PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    # Stub para evitar errores al importar desde otros módulos
    class nn:  # noqa
        Module = object
    F = None


ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class SmallCNN(nn.Module if TORCH_AVAILABLE else object):
    """
    CNN pequeña para clasificar un carácter entre len(ALPHABET) clases.

    Si PyTorch no está disponible:
        - la clase hereda de object y __init__ termina sin inicializar capas.
        - esto permite que el resto del código cargue un "stub" sin fallar.
    """

    def __init__(self, n_classes: int = len(ALPHABET)):
        if not TORCH_AVAILABLE:
            # Se permite instanciar la clase sin PyTorch (stub).
            return

        super().__init__()

        # Bloque convolucional
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2)

        # Regularización
        self.dropout = nn.Dropout(0.25)

        # Clasificador totalmente conectado
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        """
        Forward pass clásico:
        Conv → ReLU → Pool → Conv → ReLU → Pool → Conv → ReLU → Pool →
        Dropout → Flatten → Dense → ReLU → Dense
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("SmallCNN: PyTorch no está disponible.")

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.dropout(x)

        # Flatten manteniendo batch size
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x
