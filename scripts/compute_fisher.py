"""
EATA için Kaynak Doğrulama Kümesi Üzerinde Fisher Bilgi Matrisi Köşegeni Hesaplama
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.resnet import get_cifar_resnet18
from src.methods.bn_adapt import BlendedBatchNorm2d, convert_to_blended_bn
from src.methods.tent import configure_tent_model, collect_tent_params
from src.data.cifar10 import get_cifar10_dataloaders


def compute_fisher_diagonal(
    checkpoint_path: str = "src/models/checkpoints/resnet18_cifar10_seed42.pth",
    save_path: str = "src/models/checkpoints/fisher_cifar10_seed42.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    num_samples: int = 2000,
) -> dict:
    print("=" * 70)
    print("EATA İÇİN FISHER BİLGİ DİYAGONALİ HESAPLANIYOR")
    print(f"Model: {checkpoint_path} | Örnek Sayısı: {num_samples}")
    print("=" * 70)

    # 1. Modeli yükle ve TENT gibi BN katmanlarını BlendedBatchNorm2d yap
    model = get_cifar_resnet18(num_classes=10)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model = configure_tent_model(model, blend_lambda=0.9)
    model.eval()

    params, param_names = collect_tent_params(model)
    print(f"Optimize edilecek BN afin parametre sayısı: {len(params)} tensör")

    # 2. Kaynak doğrulama (val) kümesini al (eğitim verisi kullanılmaz, leakage yok)
    _, val_loader, _ = get_cifar10_dataloaders(data_dir="./data", batch_size=64, seed=42)

    fisher_diag = {name: torch.zeros_like(p, device=device) for name, p in zip(param_names, params)}
    total_processed = 0

    criterion = nn.CrossEntropyLoss()

    for imgs, targets in val_loader:
        if total_processed >= num_samples:
            break
        imgs, targets = imgs.to(device), targets.to(device)
        b = imgs.size(0)

        # Model tahmini üzerinden ampirik Fisher
        outputs = model(imgs)
        preds = outputs.argmax(dim=-1)

        model.zero_grad()
        loss = criterion(outputs, preds)
        loss.backward()

        for name, p in zip(param_names, params):
            if p.grad is not None:
                fisher_diag[name] += (p.grad.data ** 2) * b

        total_processed += b

    # Ortalama al ve normalize et
    for name in fisher_diag:
        fisher_diag[name] /= total_processed

    # Fisher değerlerini kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "fisher_diagonal": fisher_diag,
        "param_names": param_names,
        "num_samples": total_processed,
        "checkpoint_path": checkpoint_path,
    }, save_path)

    print(f"Fisher matrisi başarıyla kaydedildi: {save_path}")
    return fisher_diag


if __name__ == "__main__":
    compute_fisher_diagonal()

