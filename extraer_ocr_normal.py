from pathlib import Path
import shutil
import csv

# ==== CONFIGURÁ ESTO ====
SRC_DIR = Path(r"C:\Users\lautaro.iriart\Desktop\Procesamiento de imagenes digitales\imagenes")  # carpeta que contiene 719, 720, 721, ...
DST_DIR = Path(r"C:\Users\lautaro.iriart\Desktop\Procesamiento de imagenes digitales\media")                             # carpeta destino donde juntar todo
START_INDEX = 1                                                             # desde qué número arrancar
ZERO_PAD = False                                                            # True -> 0001, 0002... | False -> 1, 2...
MOVE_INSTEAD_OF_COPY = False                                                # True = mover; False = copiar
# ========================

DST_DIR.mkdir(parents=True, exist_ok=True)
mapping_path = DST_DIR / "mapping_ocr_normal.csv"

def candidate_is_ocr_normal(p: Path) -> bool:
    # nombre base exactamente "OCR_NORMAL" ignorando may/min (con o sin extensión)
    return p.is_file() and p.stem.lower() == "ocr_placa"

# 1) contar cuántos hay para decidir el padding (opcional)
all_candidates = []
for sub in SRC_DIR.iterdir():
    if sub.is_dir():
        files = [f for f in sub.iterdir() if candidate_is_ocr_normal(f)]
        if files:
            # si hay más de uno, agarramos el primero
            all_candidates.append(files[0])

total = len(all_candidates)
width = len(str(total + START_INDEX - 1))
if not ZERO_PAD:
    width = 0  # sin padding

# 2) copiar/mover y renombrar
i = START_INDEX
with mapping_path.open("w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["origen", "nuevo_nombre"])
    for src_file in sorted(all_candidates):
        ext = src_file.suffix  # conserva .jpg/.png/etc.
        new_name = (f"{i:0{width}d}" if width else f"{i}") + ext
        dst_file = DST_DIR / new_name

        if MOVE_INSTEAD_OF_COPY:
            shutil.move(str(src_file), dst_file)
        else:
            shutil.copy2(src_file, dst_file)

        writer.writerow([str(src_file), new_name])
        print(f"OK -> {src_file}  =>  {dst_file}")
        i += 1

print(f"\nListo. Archivos en: {DST_DIR}")
print(f"CSV de trazabilidad: {mapping_path}")
