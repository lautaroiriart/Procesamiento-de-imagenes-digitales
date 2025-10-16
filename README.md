
# 🚗 TFI 2025 – ALPR MERCOSUR (Django + PDI + CNN desde Cero)

Proyecto final con **Django** como framework web y un **OCR CNN** entrenado **desde cero**.
- Detección/segmentación con **OpenCV**.
- OCR con **PyTorch** (sin modelos pre-entrenados).
- Comando `manage.py train_ocr` para entrenar con **datos sintéticos**.

## 🗂️ Estructura
```
tfi_pdi_alpr/
├─ manage.py
├─ requirements.txt
├─ tfi_pdi_alpr/                 # settings/urls/asgi/wsgi
├─ alpr/
│  ├─ urls.py / views.py
│  ├─ templates/alpr/upload.html # demo UI
│  └─ ml/                        # PDI + OCR
│     ├─ detect.py / rectify.py / segment.py / postprocess.py
│     ├─ ocr_net.py (CNN) / ocr_model.py (loader)
│     └─ inference.py (pipeline end-to-end)
│  └─ management/commands/train_ocr.py
├─ models/                       # pesos entrenados (gitignored)
├─ data/                         # datasets (gitignored)
└─ docs/images/
```

## ⚙️ Instalación
```bash
python -m venv .venv && source .venv/bin/activate   # Win: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# Instalar PyTorch según tu SO: https://pytorch.org/get-started/locally/
# Ejemplo CPU:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python manage.py migrate
python manage.py runserver
# Abrí http://127.0.0.1:8000/alpr/upload/
```

## 🧪 Entrenamiento del OCR (sintético)
```bash
python manage.py train_ocr --epochs 5 --train-samples 8000 --val-samples 1000
# Guardará modelos en: models/ocr_cnn.pt
```
La inferencia usará ese archivo automáticamente si existe; si no, caerá a un **stub**.

## 📸 Demo Web
- Subí una imagen en `/alpr/upload/`
- Muestra JSON con `plate_text`, `bbox` y `per_char_conf`

## 🧠 Notas técnicas
- El entrenamiento sintético renderiza caracteres (A–Z, 0–9) con PIL y aplica rotación/ruido/blur.
- Luego, con datos reales, reemplazá el dataset sintético por caracteres segmentados de placas reales para afinar el modelo.

## ✅ Roadmap corto
1. Recolectar placas MERCOSUR reales → `data/raw/`
2. Extraer caracteres reales (con `ml/segment.py`) → `data/processed/chars/`
3. Reentrenar CNN con datos reales
4. Evaluación y métricas en el informe
