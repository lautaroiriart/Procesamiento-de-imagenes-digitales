🚗 TFI 2025 – ALPR MERCOSUR (Django + PDI + CNN desde cero)

Proyecto final integrador: Reconocimiento Automático de Patentes (ALPR) para formato MERCOSUR.
Pipeline mixto PDI clásico (OpenCV) + OCR con CNN entrenada desde cero (sin modelos preentrenados).
Incluye demo web en Django, comando de entrenamiento sintético, y script para preparar dataset real de caracteres.

Diagrama del pipeline: docs/images/pipeline_diagram.png
Muestra el flujo: Detección → Rectificación → Segmentación → OCR CNN → Postproceso → JSON.

🗂️ Estructura del proyecto
tfi_pdi_alpr/
├─ manage.py
├─ requirements.txt
├─ .gitignore
├─ tfi_pdi_alpr/                 # Settings/URLs/WSGI/ASGI
│  ├─ settings.py                # OCR_WEIGHTS a models/ocr_cnn.pt
│  └─ urls.py                    # /alpr/
├─ alpr/                         # App principal
│  ├─ urls.py                    # /alpr/upload/
│  ├─ views.py                   # vista de demo (subir imagen → JSON)
│  ├─ templates/alpr/upload.html # HTML “semi lindo” para demo
│  ├─ ml/                        # 🔥 PDI + OCR
│  │   ├─ detect.py              # Detección de placa (Canny, contornos, heurísticas)
│  │   ├─ rectify.py             # Recorte + resize placa (256×64)
│  │   ├─ segment.py             # Segmentación de caracteres → 32×32
│  │   ├─ postprocess.py         # Correcciones O→0, I→1, regex de formato
│  │   ├─ ocr_net.py             # SmallCNN (PyTorch) – 1 char 32×32 → clase
│  │   ├─ ocr_model.py           # Loader de pesos (usa settings.OCR_WEIGHTS) + stub
│  │   └─ inference.py           # Orquestador end-to-end (cachea el modelo)
│  └─ management/commands/
│      └─ train_ocr.py           # Entrena CNN con datos sintéticos
├─ scripts/
│  └─ prepare_chars.py           # Extrae chars 32×32 desde placas (dataset real)
├─ data/                         # datasets (gitignored)
│  ├─ raw/       # originales
│  ├─ interim/   # intermedios (ej: plates/)
│  └─ processed/ # chars/ + labels.csv para entrenamiento real
├─ models/                       # ocr_cnn.pt (pesos entrenados, gitignored)
└─ docs/images/pipeline_diagram.png

🧰 Requisitos

Python 3.12 (recomendado)

En Windows, con Python 3.14 hay problemas de wheels de opencv-python/numpy.
Bajá 3.12 x64: https://www.python.org/downloads/release/python-3120/
 (marcar Add Python to PATH).

Sistema operativo: Windows 10/11, Linux o macOS.

⚙️ Instalación (paso a paso)

Todos los comandos se corren desde la carpeta del proyecto (donde está manage.py).
Los comandos usan PowerShell (Windows) o bash (Linux/macOS).

1) Crear entorno virtual (Python 3.12)

Windows (PowerShell):

py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -V    # debe decir 3.12.x


Linux/macOS:

python3.12 -m venv .venv
source .venv/bin/activate
python -V

2) Instalar dependencias del proyecto
python -m pip install -U pip setuptools wheel
python -m pip install Django "numpy<2.3" opencv-python pillow scikit-image matplotlib


Si opencv-python te falla, probá la variante sin GUI:

python -m pip install opencv-python-headless

3) Instalar PyTorch (CPU)
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu


Si querés GPU (CUDA), ver opciones en: https://pytorch.org/get-started/locally/

▶️ Ejecutar la app web
python manage.py migrate
python manage.py runserver


Abrí el navegador: http://127.0.0.1:8000/alpr/upload/

Subí una imagen y vas a ver un JSON con:

plate_text: texto estimado (o stub si no hay modelo entrenado)

per_char_conf: confianzas por carácter

bbox: bounding box de la placa en la imagen original

🧠 Cómo funciona el modelo (resumen para explicar)

Detección (OpenCV): gris → suavizado → Canny → contornos → heurística de ratio (ancho/alto) → bbox.

Rectificación: recorte de bbox y resize a 256×64.

Segmentación: binarización adaptativa, componentes conectados, filtros por tamaño/ratio, orden izquierda→derecha, resize a 32×32 por carácter.

OCR CNN (SmallCNN): 3 conv + maxpool + dropout + 2 FC → 36 clases (0–9, A–Z).

Postproceso: reemplazos comunes (O→0, I→1, B→8) y regex de formatos MERCOSUR.

Importante: si no existe models/ocr_cnn.pt, el loader cae a un stub que devuelve "X" con confianza 0.5 para sostener la demo.

🏋️ Entrenamiento del OCR (datos sintéticos)

Sirve como bootstrap para que la CNN aprenda formas básicas de caracteres y la demo funcione sin necesitar fotos reales.

python manage.py train_ocr --epochs 5 --train-samples 8000 --val-samples 1000


Genera caracteres con PIL (A–Z, 0–9) + ruido/blur/rotación (data augmentation simple).

Entrena SmallCNN desde cero con Adam.

Guarda el mejor checkpoint en models/ocr_cnn.pt.

La inferencia lo toma automático (ruta configurada en settings.py como OCR_WEIGHTS).

🧪 Preparar dataset real de caracteres (cuando tengas placas)

Colocá placas recortadas en:

data/interim/plates/
  ├─ placa_001.jpg
  ├─ placa_002.png
  └─ ...


(Opcional) Ground-truth por imagen:

data/interim/plates.csv
filename,plate_text
placa_001.jpg,ABC1D23
placa_002.png,AB123CD


Ejecutar el script:

python scripts/prepare_chars.py \
  --plates-dir data/interim/plates \
  --out-dir data/processed/chars \
  --labels-csv data/interim/plates.csv \
  --visual-check


Qué genera:

data/processed/chars/{A..Z,0..9}/*.png (32×32 etiquetados)

data/processed/chars/_unlabeled/*.png (si no hubo GT o no coincide el largo)

data/processed/chars/labels.csv y labels_unlabeled.csv

data/processed/chars/_mosaics/*.png (tiras visuales para revisar segmentación)

Siguiente paso: adaptar el comando de entrenamiento para leer labels.csv (misma arquitectura).
(Lo podemos agregar como train_ocr_real.py si lo necesitan.)

🧩 Variables importantes

Ruta de pesos del OCR: en tfi_pdi_alpr/settings.py

OCR_WEIGHTS = str(BASE_DIR / "models" / "ocr_cnn.pt")


Cambiá esta ruta si querés cargar otros pesos.

🧯 Troubleshooting (Windows)

Error: ModuleNotFoundError: No module named 'cv2'
👉 Solución: python -m pip install opencv-python (o opencv-python-headless).

Error instalando opencv/numpy en Python 3.14:
👉 Usar Python 3.12. Crear venv nuevo:

# PowerShell
deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip setuptools wheel
python -m pip install Django "numpy<2.3" opencv-python pillow scikit-image matplotlib


pip no anda
👉 Usá python -m pip install <paquete> (dentro del venv).

Verificar versiones:

python -c "import cv2, numpy, torch; print(cv2.__version__, numpy.__version__, torch.__version__)"


PowerShell vs CMD

Borrar .venv en PowerShell:

Remove-Item -Recurse -Force .venv


(No usar rmdir /S /Q .venv en PowerShell, es de CMD.)


🧑‍💻 Comandos útiles (copiar y pegar)
# 1) Crear venv (3.12) + activar
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# 2) Instalar proyecto + OpenCV + Torch (CPU)
python -m pip install -U pip setuptools wheel
python -m pip install Django "numpy<2.3" opencv-python pillow scikit-image matplotlib
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3) Migrar + correr
python manage.py migrate
python manage.py runserver
# http://127.0.0.1:8000/alpr/upload/

# 4) Entrenar OCR (sintético)
python manage.py train_ocr --epochs 5 --train-samples 8000 --val-samples 1000

# 5) Preparar dataset real de caracteres 
python scripts/prepare_chars.py --plates-dir data/interim/plates --out-dir data/processed/chars --labels-csv data/interim/plates.csv --visual-check
