from alpr.ocr_model import load_model, predict_chars
import numpy as np

def test_stub_predict():
    m = load_model()
    x = [np.zeros((32,32), dtype=np.uint8) for _ in range(3)]
    preds, confs = predict_chars(m, x)
    assert len(preds) == 3 and len(confs) == 3
