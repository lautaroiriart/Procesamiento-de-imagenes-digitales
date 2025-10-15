"""Entrenamiento de CNN OCR desde cero con datos sintéticos (estructura lista)."""
import argparse, random, pathlib, numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F
    import torch.optim as optim
    from alpr.ocr_net import SmallCNN, ALPHABET
except Exception as e:
    raise RuntimeError("Este script requiere PyTorch instalado. Ver README para instrucciones.") from e

def _load_font(font_path: str | None, size: int = 28):
    if font_path and pathlib.Path(font_path).exists():
        return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()

def render_char(c: str, img_size: int = 32, font=None):
    if font is None:
        font = _load_font(None, size=28)
    canvas = Image.new("L", (img_size, img_size), color=0)
    draw = ImageDraw.Draw(canvas)
    try:
        bbox = draw.textbbox((0,0), c, font=font)
        w, h = bbox[2]-bbox[0], bbox[3]-bbox[1]
    except Exception:
        w, h = draw.textsize(c, font=font)
    x = (img_size - w) // 2 + random.randint(-1, 1)
    y = (img_size - h) // 2 + random.randint(-1, 1)
    draw.text((x, y), c, font=font, fill=255)
    angle = random.uniform(-6, 6)
    canvas = canvas.rotate(angle, resample=Image.BILINEAR, expand=False, fillcolor=0)
    arr = np.array(canvas)
    if random.random() < 0.3:
        arr = cv2.GaussianBlur(arr, (3,3), 0)
    if random.random() < 0.3:
        noise = np.random.normal(0, 10, arr.shape).astype(np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    arr = cv2.resize(arr, (32,32), interpolation=cv2.INTER_AREA)
    return arr

class SyntheticCharDataset(Dataset):
    def __init__(self, n_samples: int = 5000, font_path: str | None = None, seed: int = 42):
        self.n_samples = n_samples
        self.font = _load_font(font_path, size=28)
        random.seed(seed)
        self.labels = [random.choice(ALPHABET) for _ in range(n_samples)]
    def __len__(self): return self.n_samples
    def __getitem__(self, idx):
        c = self.labels[idx]
        img = render_char(c, img_size=32, font=self.font)
        x = (img.astype(np.float32) / 255.0)[None, :, :]
        y = ALPHABET.index(c)
        import torch
        return torch.tensor(x), torch.tensor(y, dtype=torch.long)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = SmallCNN(n_classes=len(ALPHABET))
    model.to(device)
    train_ds = SyntheticCharDataset(n_samples=args.train_samples, font_path=args.font)
    val_ds   = SyntheticCharDataset(n_samples=args.val_samples,   font_path=args.font, seed=7)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    best_val = 0.0
    out_path = pathlib.Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum, n = 0.0, 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
        train_loss = loss_sum / max(n, 1)
        model.eval()
        correct, tot = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                tot += y.numel()
        val_acc = correct / max(tot, 1)
        print(f"Epoch {epoch}/{args.epochs} - train_loss: {train_loss:.4f}  val_acc: {val_acc:.3f}")
        if val_acc > best_val:
            best_val = val_acc
            torch.save(model.state_dict(), out_path)
            print(f"✔ Guardado mejor modelo en {out_path} (val_acc={val_acc:.3f})")
    print(f"Entrenamiento finalizado. Mejor val_acc: {best_val:.3f}")

def parse_args():
    import argparse
    p = argparse.ArgumentParser(description="Entrenamiento CNN OCR desde cero (datos sintéticos)")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--train-samples", type=int, default=8000)
    p.add_argument("--val-samples", type=int, default=1000)
    p.add_argument("--font", type=str, default=None, help="Ruta a .ttf opcional (estilo MERCOSUR)")
    p.add_argument("--out", type=str, default="models/ocr_cnn.pt")
    p.add_argument("--cpu", action="store_true", help="Forzar CPU")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args)
