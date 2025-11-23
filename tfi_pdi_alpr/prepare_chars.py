#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Extrae caracteres 32x32 desde placas recortadas para construir datasets.
Uso típico:

    python scripts/prepare_chars.py \
        --plates-dir data/interim/plates \
        --out-dir data/processed/chars \
        --labels-csv data/interim/plates.csv

El CSV opcional debe contener:
    filename, plate_text
"""

import argparse
import csv
import glob
import sys
from pathlib import Path

import cv2
import numpy as np

# Permite importar módulos del proyecto cuando el script se ejecuta desde CLI
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpr.ml.segment import split_characters
from alpr.ml.ocr_net import ALPHABET


# ----------------------------------------------------------
# Utilidades auxiliares
# ----------------------------------------------------------

def ensure_dir(path: Path) -> None:
    """Crea un directorio si no existe."""
    path.mkdir(parents=True, exist_ok=True)


def load_ground_truth_map(csv_path: Path) -> dict[str, str]:
    """
    Carga un CSV opcional con mapeo filename → plate_text.

    Retorna:
        dict con claves = nombre de archivo, valor = texto de patente.
    """
    if not csv_path or not csv_path.exists():
        return {}

    mapping = {}

    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename")
            text = row.get("plate_text")
            if filename and text:
                mapping[filename] = text.strip().upper()

    return mapping


def find_image_files(directory: Path) -> list[Path]:
    """Busca imágenes en formatos comunes dentro de un directorio."""
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif")
    files = []

    for ext in extensions:
        files.extend(glob.glob(str(directory / ext)))

    return sorted(Path(fp) for fp in files)


def save_character(
    output_dir: Path,
    label: str,
    image: np.ndarray,
    base_name: str,
    index: int,
) -> str:
    """
    Guarda un carácter recortado en la carpeta correspondiente a su clase.

    Retorna:
        nombre del archivo generado.
    """
    ensure_dir(output_dir / label)
    filename = f"{base_name}_{index}.png"
    full_path = output_dir / label / filename
    cv2.imwrite(str(full_path), image)
    return filename


def save_mosaic(characters: list[np.ndarray], output_path: Path) -> None:
    """Genera un mosaico horizontal para inspección visual."""
    if not characters:
        return

    gap = 4
    h, w = characters[0].shape[:2]
    canvas = np.zeros((h, len(characters) * (w + gap) - gap), dtype=np.uint8)

    for i, ch in enumerate(characters):
        x = i * (w + gap)
        canvas[:, x:x + w] = ch

    ensure_dir(output_path.parent)
    cv2.imwrite(str(output_path), canvas)


# ----------------------------------------------------------
# Lógica principal
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Extrae caracteres 32x32 desde placas recortadas."
    )
    parser.add_argument(
        "--plates-dir",
        type=str,
        default="data/interim/plates",
        help="Directorio que contiene placas recortadas",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="data/processed/chars",
        help="Directorio de salida (una carpeta por clase)",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="CSV opcional con filename y plate_text",
    )
    parser.add_argument(
        "--visual-check",
        action="store_true",
        help="Guardar mosaicos de caracteres para inspección visual",
    )
    args = parser.parse_args()

    plates_dir = Path(args.plates_dir)
    output_dir = Path(args.out_dir)

    ensure_dir(output_dir)

    # Mapa filename → plate_text en mayúsculas
    gt_map = load_ground_truth_map(Path(args.labels_csv)) if args.labels_csv else {}

    # Donde se registran muestras
    labeled_samples = []
    unlabeled_samples = []

    # Buscar imágenes
    image_files = find_image_files(plates_dir)
    if not image_files:
        print(f"[WARN] No se encontraron imágenes en {plates_dir}")
        return

    # Crear carpetas por clase
    for label in list(ALPHABET) + ["_unlabeled"]:
        ensure_dir(output_dir / label)

    mosaics_dir = output_dir / "_mosaics"

    # Procesar cada placa
    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] No se pudo leer {image_path}")
            continue

        # Convertir a RGB para segmentación
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        characters = split_characters(image_rgb)

        if not characters:
            print(f"[INFO] Sin caracteres detectados en {image_path.name}")
            continue

        gt_text = gt_map.get(image_path.name)
        base = image_path.stem

        # Con ground truth válido
        if gt_text and len(gt_text) == len(characters):
            for i, char_img in enumerate(characters):
                label = gt_text[i] if gt_text[i] in ALPHABET else "_unlabeled"
                filename = save_character(output_dir, label, char_img, base, i)
                labeled_samples.append({"filename": filename, "label": label})

        # Sin ground truth (o longitud inconsistente)
        else:
            for i, char_img in enumerate(characters):
                filename = save_character(output_dir, "_unlabeled", char_img, base, i)
                unlabeled_samples.append({"filename": filename, "label": "_unlabeled"})

        # Guardar mosaico opcional
        if args.visual_check:
            mosaic_path = mosaics_dir / f"{base}_mosaic.png"
            save_mosaic(characters, mosaic_path)

    # Guardar CSVs finales
    if labeled_samples:
        with (output_dir / "labels.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(labeled_samples)
        print(f"[OK] labels.csv guardado con {len(labeled_samples)} filas")

    if unlabeled_samples:
        with (output_dir / "labels_unlabeled.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "label"])
            writer.writeheader()
            writer.writerows(unlabeled_samples)
        print(
            f"[OK] labels_unlabeled.csv guardado con {len(unlabeled_samples)} filas"
        )

    print("[DONE] Preparación de caracteres finalizada.")


if __name__ == "__main__":
    main()
