#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Normaliza nombres de imágenes provenientes de múltiples carpetas, copiándolas
(o moviéndolas) a un único directorio unificado. También genera un CSV de
trazabilidad origen → nuevo_nombre.

Funciona con estructuras del estilo:
    SRC_DIR/
        719/
        720/
        721/
        ...

Cada subcarpeta contiene un archivo llamado OCR_PLACA.* que se debe renombrar
y copiar/mover al directorio destino.
"""

from pathlib import Path
import csv
import shutil


# ---------------------------------------------------------
# Configuración del usuario
# ---------------------------------------------------------

SRC_DIR = Path(
    r"C:\Users\lautaro.iriart\Desktop\Procesamiento de imagenes digitales\imagenes"
)
DST_DIR = Path(
    r"C:\Users\lautaro.iriart\Desktop\Procesamiento de imagenes digitales\media"
)

START_INDEX = 1
ZERO_PAD = False
MOVE_INSTEAD_OF_COPY = False


# ---------------------------------------------------------
# Utilidades
# ---------------------------------------------------------

def is_plate_image(file_path: Path) -> bool:
    """
    Verifica si el archivo corresponde al patrón 'ocr_placa.xxx'.

    Ignora mayúsculas/minúsculas y acepta cualquier extensión.
    """
    return file_path.is_file() and file_path.stem.lower() == "ocr_placa"


def get_candidate_files(source_dir: Path) -> list[Path]:
    """
    Recorre las subcarpetas de source_dir buscando un único archivo OCR_PLACA.*
    en cada una.

    Retorna una lista con las rutas encontradas.
    """
    candidates = []

    for sub in source_dir.iterdir():
        if sub.is_dir():
            matches = [f for f in sub.iterdir() if is_plate_image(f)]
            if matches:
                candidates.append(matches[0])

    return sorted(candidates)


def compute_padding_width(count: int, start_index: int, zero_pad: bool) -> int:
    """
    Determina cuántos dígitos usar para el padding numérico.
    Si zero_pad es False, retorna 0 para desactivar padding.
    """
    if not zero_pad:
        return 0

    max_value = count + start_index - 1
    return len(str(max_value))


def save_traceability_csv(path: Path, rows: list[tuple[str, str]]):
    """Guarda el CSV origen → nuevo_nombre."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["origen", "nuevo_nombre"])
        writer.writerows(rows)


def rename_and_transfer(
    candidates: list[Path],
    dst_dir: Path,
    start_index: int,
    pad_width: int,
    move: bool,
) -> list[tuple[str, str]]:
    """
    Copia o mueve los archivos encontrados a dst_dir, asignando nombres nuevos.
    Retorna una lista de pares (origen, nuevo_nombre) para el CSV.
    """
    results = []
    counter = start_index

    for src_file in candidates:
        ext = src_file.suffix
        new_name = (
            f"{counter:0{pad_width}d}" if pad_width else str(counter)
        ) + ext

        dst_file = dst_dir / new_name

        if move:
            shutil.move(str(src_file), dst_file)
        else:
            shutil.copy2(src_file, dst_file)

        results.append((str(src_file), new_name))
        print(f"[OK] {src_file} → {dst_file}")

        counter += 1

    return results


# ---------------------------------------------------------
# Ejecución principal
# ---------------------------------------------------------

def main():
    DST_DIR.mkdir(parents=True, exist_ok=True)
    mapping_csv = DST_DIR / "mapping_ocr_normal.csv"

    candidates = get_candidate_files(SRC_DIR)
    if not candidates:
        print(f"[WARN] No se encontraron archivos OCR_PLACA.* en {SRC_DIR}")
        return

    pad_width = compute_padding_width(
        count=len(candidates),
        start_index=START_INDEX,
        zero_pad=ZERO_PAD,
    )

    trace_rows = rename_and_transfer(
        candidates=candidates,
        dst_dir=DST_DIR,
        start_index=START_INDEX,
        pad_width=pad_width,
        move=MOVE_INSTEAD_OF_COPY,
    )

    save_traceability_csv(mapping_csv, trace_rows)

    print("\nListo.")
    print(f"Archivos normalizados en: {DST_DIR}")
    print(f"CSV de trazabilidad: {mapping_csv}")


if __name__ == "__main__":
    main()
