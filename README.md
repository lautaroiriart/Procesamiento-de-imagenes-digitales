# TFI – ALPR MERCOSUR (PDI + CNN desde cero)

Pipeline clásico de PDI para detectar y segmentar la patente + OCR con CNN **entrenada desde cero** (sin modelos pre-entrenados).

## Ejecutar API rápida
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
# (Opcional) Instalar PyTorch según tu sistema: https://pytorch.org/get-started/locally/
uvicorn api.main:app --reload --port 8000
```
Abrí http://localhost:8000/docs y probá `/health`.

## Estructura
- `api/` FastAPI + servicio de inferencia
- `alpr/` módulos de detección, rectificación, segmentación, modelo y postproceso
- `scripts/` utilidades para descargar y preparar datos
- `tests/` pruebas rápidas
- `frontend/` HTML mínimo para probar
- `docker/` Dockerfile y nginx (opcional para producción)

## Notas
- El modelo de OCR se entrena desde cero (ver `alpr/ocr_model.py` y `alpr/ocr_train.py` **(TODO)**).
- Para mañana: API funcional con `/health` y `/predict` (stub), demo de detección básica y plan de entrenamiento.
