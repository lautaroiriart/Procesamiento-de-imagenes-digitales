from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from . import version
from alpr.inference_service import infer_image

app = FastAPI(title="TFI ALPR", version=version.__version__)

class PredictOut(BaseModel):
    plate_text: str | None = None
    per_char_conf: list[float] | None = None
    bbox: list[int] | None = None

@app.get("/health")
def health():
    return {"status": "ok", "version": version.__version__}

@app.post("/predict", response_model=PredictOut)
async def predict(file: UploadFile = File(...)):
    img_bytes = await file.read()
    result = infer_image(img_bytes)
    return result
