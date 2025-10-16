
from django.core.management.base import BaseCommand
from django.conf import settings
from pathlib import Path
import random, numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    import torch.optim as optim
    from alpr.ml.ocr_net import SmallCNN, ALPHABET
except Exception as e:
    raise RuntimeError("Instalá PyTorch para entrenar el OCR (ver README).") from e

def _font(font_path: str | None, size: int = 28):
    if font_path and Path(font_path).exists():
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()

def _render_char(c: str, size=32, font=None):
    if font is None: font = _font(None, 28)
    canvas = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(canvas)
    try:
        bbox = draw.textbbox((0,0), c, font=font); w,h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        w,h = draw.textsize(c, font=font)
    import random as R
    x = (size - w)//2 + R.randint(-1,1)
    y = (size - h)//2 + R.randint(-1,1)
    draw.text((x,y), c, font=font, fill=255)
    angle = R.uniform(-6,6)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    arr = np.array(canvas)
    if R.random() < 0.3: arr = cv2.GaussianBlur(arr,(3,3),0)
    if R.random() < 0.3:
        noise = np.random.normal(0,10,arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16)+noise,0,255).astype(np.uint8)
    return cv2.resize(arr,(32,32), interpolation=cv2.INTER_AREA)

class SynthDS(Dataset):
    def __init__(self, n=8000, font_path=None, seed=42):
        self.n=n; self.font=_font(font_path,28)
        random.seed(seed); self.labels=[random.choice(ALPHABET) for _ in range(n)]
    def __len__(self): return self.n
    def __getitem__(self, i):
        c=self.labels[i]; img=_render_char(c,32,self.font)
        x=(img.astype(np.float32)/255.0)[None,:,:]
        y=ALPHABET.index(c)
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

class Command(BaseCommand):
    help = "Entrena la CNN OCR desde cero con datos sintéticos."

    def add_arguments(self, parser):
        parser.add_argument("--epochs", type=int, default=5)
        parser.add_argument("--batch-size", type=int, default=128)
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--train-samples", type=int, default=8000)
        parser.add_argument("--val-samples", type=int, default=1000)
        parser.add_argument("--font", type=str, default=None)
        parser.add_argument("--out", type=str, default=None)
        parser.add_argument("--cpu", action="store_true")

    def handle(self, *args, **opts):
        device = torch.device("cuda" if torch.cuda.is_available() and not opts["cpu"] else "cpu")
        model = SmallCNN().to(device)

        train_ds = SynthDS(opts["train_samples"], opts["font"])
        val_ds   = SynthDS(opts["val_samples"],   opts["font"], seed=7)
        train_dl = DataLoader(train_ds, batch_size=opts["batch_size"], shuffle=True)
        val_dl   = DataLoader(val_ds,   batch_size=opts["batch_size"], shuffle=False)

        opt = optim.Adam(model.parameters(), lr=opts["lr"])
        best=0.0
        out_path = Path(opts["out"] or getattr(settings,"OCR_WEIGHTS", Path("models/ocr_cnn.pt")))
        out_path.parent.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, opts["epochs"]+1):
            model.train(); loss_sum=0; n=0
            for x,y in train_dl:
                x,y = x.to(device), y.to(device)
                loss = F.cross_entropy(model(x), y)
                opt.zero_grad(); loss.backward(); opt.step()
                loss_sum += loss.item()*x.size(0); n += x.size(0)
            tr_loss = loss_sum/max(n,1)

            model.eval(); corr=tot=0
            with torch.no_grad():
                for x,y in val_dl:
                    x,y = x.to(device), y.to(device)
                    pred = model(x).argmax(1)
                    corr += (pred==y).sum().item(); tot += y.numel()
            val = corr/max(tot,1)
            self.stdout.write(f"Epoch {epoch}/{opts['epochs']} - loss {tr_loss:.4f} - val_acc {val:.3f}")
            if val>best:
                best=val; torch.save(model.state_dict(), out_path)
                self.stdout.write(self.style.SUCCESS(f"Guardado: {out_path} (val={val:.3f})"))
        self.stdout.write(self.style.SUCCESS(f"Listo. Mejor val_acc={best:.3f}"))
