# alpr/management/commands/train_ocr.py

from pathlib import Path

import numpy as np
from django.conf import settings
from django.core.management.base import BaseCommand
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

from alpr.ml.dataset import load_plate_dataset, IMG_HEIGHT, IMG_WIDTH
from alpr.ml.encoding import encode_text, ALPHABET, MAX_LEN


def build_ocr_model(input_shape, num_classes, sequence_length):
    """
    CNN para reconocimiento carácter-por-carácter en patentes.
    Cada posición de la secuencia tiene su propia salida softmax.
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

    outputs = [
        layers.Dense(num_classes, activation="softmax", name=f"char_{i}")(x)
        for i in range(sequence_length)
    ]

    model = models.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"] * sequence_length,
    )
    return model


def prepare_labels(texts):
    """
    Codifica cada patente como una secuencia de índices.
    Retorna un diccionario con un vector por salida del modelo.
    """
    encoded_sequences = np.array([encode_text(t) for t in texts])

    label_dict = {
        f"char_{i}": encoded_sequences[:, i] for i in range(MAX_LEN)
    }
    return encoded_sequences, label_dict


class Command(BaseCommand):
    help = "Entrena el modelo OCR basado en CNN utilizando el dataset de patentes."

    def handle(self, *args, **options):

        # --- Carga de datos ---
        images, plate_texts = load_plate_dataset()
        total_samples = len(plate_texts)
        self.stdout.write(self.style.SUCCESS(f"Muestras totales: {total_samples}"))

        encoded_sequences, _ = prepare_labels(plate_texts)

        # --- División en train / validation ---
        (
            X_train,
            X_val,
            seq_train,
            seq_val,
        ) = train_test_split(
            images,
            encoded_sequences,
            test_size=0.2,
            random_state=42,
        )

        # Cada salida del modelo necesita su propio vector
        y_train = {f"char_{i}": seq_train[:, i] for i in range(MAX_LEN)}
        y_val = {f"char_{i}": seq_val[:, i] for i in range(MAX_LEN)}

        num_classes = len(ALPHABET)

        # --- Construcción del modelo ---
        model = build_ocr_model(
            input_shape=(IMG_HEIGHT, IMG_WIDTH, 1),
            num_classes=num_classes,
            sequence_length=MAX_LEN,
        )
        model.summary(print_fn=self.stdout.write)

        # --- Entrenamiento ---
        model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=40,
            batch_size=16,
        )

        # --- Guardado del modelo ---
        output_dir = Path(settings.BASE_DIR) / "models"
        output_dir.mkdir(exist_ok=True)
        model_path = output_dir / "plate_ocr_model.h5"
        model.save(model_path)

        self.stdout.write(self.style.SUCCESS(f"Modelo guardado en: {model_path}"))
