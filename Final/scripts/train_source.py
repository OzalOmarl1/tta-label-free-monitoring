"""
Kaynak ResNet-18 eğitimi (CIFAR-10) — orijinal src/train_source.py'nin düzeltilmiş sürümü.

Düzeltmeler (bkz. AUDIT.md):
  * Tohum komut satırından verilir (orijinal betikte seed=42 sabit kodlanmıştı; seed 43/44
    modelleri depoda bulunmayan, sürümlenmemiş bir değişiklikle üretilmişti).
  * Log dosyası tohum ekiyle yazılır (orijinali her koşuda aynı dosyanın üzerine yazıyordu).
  * Python `random`, NumPy, Torch ve DataLoader işçi tohumları sabitlenir.
  * Termal koruyucu: her mini-batch arasında `guard.throttle()` (yalnızca duraklatma/uyku;
    batch size DEĞİŞTİRİLMEZ, böylece eğitim protokolü termal olaylardan etkilenmez).

Protokol (orijinal ile özdeş): SGD(lr=0.1, momentum=0.9, wd=5e-4), Cosine LR, 50 epok,
batch 128, RandomCrop(32,4)+HFlip, 45k/5k stratified ayrım (random_state=seed),
en iyi doğrulama doğruluğuna sahip checkpoint saklanır ve 10k temiz testte raporlanır.

Veri hattı notu: Orijinal betik PNG çözme + augmentasyonu CPU'da (DataLoader, 2 işçi) yapıyordu; bu dizüstü
bilgisayarda CPU'yu sürekli 85 °C sınırına taşıyıp eğitimi ~3 kat yavaşlattı. Bu sürümde aynı augmentasyon
(4 piksel sıfır dolgulu RandomCrop(32) + p=0.5 yatay çevirme) ve aynı normalizasyon GPU'da, tohumlu bir CUDA
üreteciyle uygulanır. Dağılım olarak özdeştir; rastgele sayı akışı farklı olduğundan seed 42–44 ile bit düzeyinde
aynı değildir (bkz. AUDIT.md, A3).

Kullanım:  python scripts/train_source.py --seed 45
"""

import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttamon.data import load_cifar10_test, load_cifar10_train, split_indices  # noqa: E402
from ttamon.models import get_cifar_resnet18  # noqa: E402
from ttamon.paths import CKPT_DIR, LOGS_DIR  # noqa: E402
from ttamon.thermal_guard import ThermalGuard  # noqa: E402


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


MEAN = torch.tensor((0.4914, 0.4822, 0.4465)).view(1, 3, 1, 1)
STD = torch.tensor((0.2023, 0.1994, 0.2010)).view(1, 3, 1, 1)


def normalize(x_u8_nhwc, device):
    x = x_u8_nhwc.permute(0, 3, 1, 2).float().div_(255.0)
    return (x - MEAN.to(device)) / STD.to(device)


def augment(xb_u8, gen):
    """RandomCrop(32, padding=4) [sıfır dolgu, piksel uzayında] + RandomHorizontalFlip(0.5). NHWC uint8."""
    B = xb_u8.shape[0]
    dev = xb_u8.device
    xp = torch.nn.functional.pad(xb_u8, (0, 0, 4, 4, 4, 4))           # (B,40,40,3)
    oy = torch.randint(0, 9, (B,), device=dev, generator=gen)
    ox = torch.randint(0, 9, (B,), device=dev, generator=gen)
    base = torch.arange(32, device=dev)
    rows = (oy[:, None] + base)[:, :, None]
    cols = (ox[:, None] + base)[:, None, :]
    out = xp[torch.arange(B, device=dev)[:, None, None], rows, cols]   # (B,32,32,3)
    flip = torch.rand(B, device=dev, generator=gen) < 0.5
    out[flip] = out[flip].flip(2)
    return out


@torch.no_grad()
def evaluate(model, x_u8, y, device, bs=500):
    model.eval()
    correct, loss_sum = 0, 0.0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for i in range(0, len(y), bs):
        out = model(normalize(x_u8[i:i + bs], device))
        loss_sum += crit(out, y[i:i + bs]).item()
        correct += (out.argmax(1) == y[i:i + bs]).sum().item()
    return 100.0 * correct / len(y), loss_sum / len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)
    out_ckpt = os.path.join(CKPT_DIR, f"resnet18_cifar10_seed{args.seed}.pth")
    if os.path.exists(out_ckpt) and not args.overwrite:
        print(f"[DUR] {out_ckpt} zaten var. Dondurulmuş checkpoint'lerin üzerine yazılmaz (--overwrite).")
        return

    xtr_all, ytr_all = load_cifar10_train()
    tr_idx, va_idx = split_indices(ytr_all, args.seed)          # orijinal ile aynı 45k/5k ayrımı
    xte, yte = load_cifar10_test()
    to = lambda a: torch.from_numpy(a).to(device)  # noqa: E731
    x_tr, y_tr = to(xtr_all[tr_idx]), to(ytr_all[tr_idx])
    x_va, y_va = to(xtr_all[va_idx]), to(ytr_all[va_idx])
    x_te, y_te = to(xte), to(yte)
    gen = torch.Generator(device=device).manual_seed(args.seed)
    perm_gen = torch.Generator().manual_seed(args.seed)

    model = get_cifar_resnet18(10).to(device)
    opt = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    guard = ThermalGuard(log_csv=os.path.join(LOGS_DIR, f"thermal_train_seed{args.seed}.csv")).start()
    hist = {"seed": args.seed, "epochs": [], "train_loss": [], "train_acc": [], "val_loss": [],
            "val_acc": [], "lr": [], "data_pipeline": "gpu_uint8_augment"}
    best_val, best_epoch = -1.0, -1
    t0 = time.time()
    n = len(y_tr)
    try:
        for ep in range(1, args.epochs + 1):
            model.train()
            correct, loss_sum = 0, 0.0
            perm = torch.randperm(n, generator=perm_gen).to(device)
            for i in range(0, n, args.batch_size):
                guard.throttle()
                bi = perm[i:i + args.batch_size]
                x = normalize(augment(x_tr[bi], gen), device)
                y = y_tr[bi]
                opt.zero_grad(set_to_none=True)
                out = model(x)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                loss_sum += loss.item() * y.numel()
                correct += (out.argmax(1) == y).sum().item()
            sched.step()
            va_acc, va_loss = evaluate(model, x_va, y_va, device)
            hist["epochs"].append(ep)
            hist["train_loss"].append(loss_sum / n)
            hist["train_acc"].append(100.0 * correct / n)
            hist["val_loss"].append(va_loss)
            hist["val_acc"].append(va_acc)
            hist["lr"].append(sched.get_last_lr()[0])
            print(f"[seed {args.seed}] epok {ep:2d}/{args.epochs} | train {100.0*correct/n:5.2f}% | "
                  f"val {va_acc:5.2f}% | {(time.time()-t0)/60:5.1f} dk | termal {guard.level}", flush=True)
            if va_acc > best_val:
                best_val, best_epoch = va_acc, ep
                torch.save({"epoch": ep, "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": opt.state_dict(), "val_acc": va_acc,
                            "seed": args.seed}, out_ckpt)
    finally:
        guard.stop()

    ckpt = torch.load(out_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_acc, _ = evaluate(model, x_te, y_te, device)
    hist.update({"best_val_acc": best_val, "best_epoch": best_epoch, "clean_test_acc": test_acc,
                 "training_time_minutes": (time.time() - t0) / 60.0, "thermal": guard.summary(),
                 "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"})
    with open(os.path.join(LOGS_DIR, f"train_curves_seed{args.seed}.json"), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2)
    print(f"[seed {args.seed}] BİTTİ | en iyi val {best_val:.2f}% (epok {best_epoch}) | temiz test {test_acc:.2f}%")


if __name__ == "__main__":
    main()
