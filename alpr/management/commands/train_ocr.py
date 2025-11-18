# alpr/management/commands/train_ocr.py
from django.core.management.base import BaseCommand
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from django.conf import settings
from alpr.ml.dataset import load_plate_dataset, IMG_HEIGHT, IMG_WIDTH
from alpr.ml.encoding import encode_text, ALPHABET, MAX_LEN


def build_model(input_shape, num_classes, max_len):
    """
    Modelo simple tipo CNN + fully connected con una salida softmax por carácter.
    """
    inputs = layers.Input(shape=input_shape)

    x = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation="relu", padding="same")(x)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation="relu")(x)

    outputs = []
    for i in range(max_len):
        outputs.append(
            layers.Dense(num_classes, activation="softmax", name=f"char_{i}")(x)
        )

    model = models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


class Command(BaseCommand):
    help = "Entrena el modelo OCR con el dataset de docs/images + mapping_ocr_normal.xlsx"

    def handle(self, *args, **options):
        # 1) Cargar dataset
        X, texts = load_plate_dataset()
        self.stdout.write(self.style.SUCCESS(f"Total de muestras: {len(texts)}"))

        # 2) Codificar etiquetas
        y_encoded = np.array([encode_text(t) for t in texts])
        y_dict = {f"char_{i}": y_encoded[:, i] for i in range(MAX_LEN)}

        X_train, X_val, y_train, y_val = train_test_split(
            X, y_dict, test_size=0.2, random_state=42
        )

        num_classes = len(ALPHABET)

        # 3) Construir modelo
        model = build_model(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
            num_classes=num_classes,
            max_len=MAX_LEN,
        )
        model.summary(print_fn=self.stdout.write)

        # 4) Entrenar
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=10,
            batch_size=16,
        )

        # 5) Guardar
        models_dir = Path(settings.BASE_DIR) / "models"
        models_dir.mkdir(exist_ok=True)
        model_path = models_dir / "plate_ocr_model.h5"
        model.save(model_path)

        self.stdout.write(self.style.SUCCESS(f"Modelo guardado en {model_path}"))
