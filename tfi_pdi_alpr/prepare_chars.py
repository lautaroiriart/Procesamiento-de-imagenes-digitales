
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Extrae caracteres 32x32 desde placas recortadas.
# Uso: python scripts/prepare_chars.py --plates-dir data/interim/plates --out-dir data/processed/chars --labels-csv data/interim/plates.csv
# CSV opcional: columnas filename,plate_text

import argparse, sys, csv, glob
from pathlib import Path
import cv2
import numpy as np

# Permitir importar módulos de la app (ejecutar desde raíz del proyecto)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alpr.ml.segment import split_characters
from alpr.ml.ocr_net import ALPHABET

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_gt_map(csv_path: Path):
    if not csv_path or not csv_path.exists():
        return {}
    gt = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fn = row.get('filename')
            pt = row.get('plate_text')
            if fn and pt:
                gt[fn] = pt.strip().upper()
    return gt

def main():
    ap = argparse.ArgumentParser(description="Preparar dataset de caracteres desde placas recortadas")
    ap.add_argument("--plates-dir", type=str, default="data/interim/plates", help="Directorio con placas (imagenes)")
    ap.add_argument("--out-dir", type=str, default="data/processed/chars", help="Salida base (carpetas por clase)")
    ap.add_argument("--labels-csv", type=str, default=None, help="CSV opcional con filename,plate_text")
    ap.add_argument("--visual-check", action="store_true", help="Guardar mosaicos de caracteres por imagen para inspección")
    args = ap.parse_args()

    plates_dir = Path(args.plates_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    gt_map = load_gt_map(Path(args.labels_csv)) if args.labels_csv else {}
    labeled_rows, unlabeled_rows = [], []

    exts = ("*.jpg","*.jpeg","*.png","*.bmp","*.tif")
    files = []
    for e in exts:
        files.extend(glob.glob(str(plates_dir / e)))
    files = sorted(files)

    if not files:
        print(f"[WARN] No se encontraron imágenes en {plates_dir}.")
        return

    classes = list(ALPHABET) + ["_unlabeled"]
    for c in classes:
        ensure_dir(out_dir / c)

    for fp in files:
        fp = Path(fp)
        img = cv2.imread(str(fp))
        if img is None:
            print(f"[WARN] No pude leer {fp}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        chars = split_characters(img)
        if not chars:
            print(f"[INFO] Sin caracteres detectados en {fp.name}")
            continue

        gt_text = gt_map.get(fp.name)

        if gt_text and len(gt_text) == len(chars):
            for i, ch_img in enumerate(chars):
                label = gt_text[i]
                if label not in ALPHABET:
                    label = "_unlabeled"
                out_name = f"{fp.stem}_{i}.png"
                cv2.imwrite(str(out_dir / label / out_name), ch_img)
                labeled_rows.append({"filename": out_name, "label": label})
        else:
            for i, ch_img in enumerate(chars):
                out_name = f"{fp.stem}_{i}.png"
                cv2.imwrite(str(out_dir / "_unlabeled" / out_name), ch_img)
                unlabeled_rows.append({"filename": out_name, "label": "_unlabeled"})

        if args.visual_check and chars:
            gap = 4
            h, w = chars[0].shape[:2]
            canv = np.zeros((h, len(chars)*(w+gap)-gap), dtype=np.uint8)
            for i, ch in enumerate(chars):
                x = i*(w+gap)
                canv[:, x:x+w] = ch
            mosaics_dir = out_dir / "_mosaics"
            ensure_dir(mosaics_dir)
            from cv2 import imwrite as _imw
            _imw(str(mosaics_dir / f"{fp.stem}_mosaic.png"), canv)

    if labeled_rows:
        with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["filename","label"])
            wr.writeheader()
            wr.writerows(labeled_rows)
        print(f"[OK] Guardado labels.csv con {len(labeled_rows)} filas")

    if unlabeled_rows:
        with open(out_dir / "labels_unlabeled.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["filename","label"])
            wr.writeheader()
            wr.writerows(unlabeled_rows)
        print(f"[OK] Guardado labels_unlabeled.csv con {len(unlabeled_rows)} filas")

    print("[DONE] Preparación de caracteres finalizada.")

if __name__ == "__main__":
    main()

