import json, random
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from data import get_dataloaders
from model import MLP

LEARNING_RATE, EPOCHS, HIDDEN, SEED = 0.01, 100, 16, 42
OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def evaluate(model, loader, criterion, device):
    model.eval()
    total, correct, n = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            total += criterion(logits, yb).item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            n += xb.size(0)
    return total / n, correct / n

def main():
    set_seed(SEED)
    device = get_device()
    print(f"PyTorch {torch.__version__} | dispositivo: {device}")

    train_loader, val_loader, n_feat, n_cls = get_dataloaders()
    model = MLP(n_feat, HIDDEN, n_cls).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    hist = {"train_loss": [], "val_loss": [], "val_acc": []}
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = criterion(model(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running += loss.item() * xb.size(0)
            n += xb.size(0)
        vl, va = evaluate(model, val_loader, criterion, device)
        hist["train_loss"].append(running / n)
        hist["val_loss"].append(vl)
        hist["val_acc"].append(va)
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch:3d} | train {running/n:.4f} | val {vl:.4f} | acc {va:.3f}")

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "history.json").write_text(json.dumps(hist, indent=2))
    plt.plot(hist["train_loss"], label="train")
    plt.plot(hist["val_loss"], label="validación")
    plt.xlabel("Época"); plt.ylabel("Pérdida"); plt.legend()
    plt.savefig(OUT_DIR / "loss_curve.png", dpi=120)

if __name__ == "__main__":
    main()