"""
Kaynak ResNet-18 Modelinin CIFAR-10 Üzerinde Eğitilmesi Betiği

Eğitim Protokolü:
- Model: ResNet-18 (CIFAR 32x32 için uyarlanmış: 3x3 conv1, no maxpool)
- Veri Kümeleri:
  * Eğitim: 45.000 örnek (CIFAR-10 train içinden %10 stratified val ayrıldıktan sonra)
  * Kaynak Doğrulama (source-val): 5.000 örnek
  * Test (source-test): 10.000 örnek
- Hiperparametreler:
  * Optimizer: SGD (lr=0.1, momentum=0.9, weight_decay=5e-4)
  * Öğrenme Oranı Zamanlayıcısı: CosineAnnealingLR (T_max=epok sayısı)
  * Batch size: 128
  * Sabit Tohum: 42 (seed=42)
- Çıktılar:
  * results/logs/cifar10_training_curves.json (eğitim ve doğrulama kayıp/doğruluk eğrileri)
  * src/models/checkpoints/resnet18_cifar10_seed42.pth (en iyi doğrulama doğruluğuna sahip checkpoint)
  * Temiz CIFAR-10 test doğruluğu raporu.
"""

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.data.cifar10 import get_cifar10_dataloaders
from src.models.resnet import get_cifar_resnet18


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import numpy as np
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_cifar10_resnet18(
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 0.1,
    weight_decay: float = 5e-4,
    seed: int = 42,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    set_seed(seed)
    print("=" * 70)
    print("FAZ 0: RESNET-18 CIFAR-10 MODEL EĞİTİMİ (KAYNAK MODEL)")
    print(f"Cihaz: {device} ({torch.cuda.get_device_name(0) if device=='cuda' else 'CPU'})")
    print(f"Epok: {epochs}, Batch Size: {batch_size}, Başlangıç LR: {lr}, Tohum: {seed}")
    print("=" * 70)

    # Veri yükleyicileri
    train_loader, val_loader, test_loader = get_cifar10_dataloaders(
        data_dir="./data",
        batch_size=batch_size,
        num_workers=2,
        val_ratio=0.10,
        seed=seed,
    )

    model = get_cifar_resnet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        "epochs": [],
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "lr": [],
    }

    best_val_acc = 0.0
    checkpoint_dir = os.path.join("src", "models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_checkpoint_path = os.path.join(checkpoint_dir, f"resnet18_cifar10_seed{seed}.pth")

    start_train_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
            preds = outputs.argmax(dim=-1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        scheduler.step()

        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Doğrulama adımı (source-val)
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, targets)

                val_loss_sum += loss.item() * targets.size(0)
                preds = outputs.argmax(dim=-1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        val_loss = val_loss_sum / val_total
        val_acc = (val_correct / val_total) * 100.0
        current_lr = scheduler.get_last_lr()[0]

        history["epochs"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        print(
            f"Epok {epoch:2d}/{epochs:2d} | "
            f"Train Kaybı: {train_loss:.4f}, Train Doğruluğu: %{train_acc:5.2f} | "
            f"Val Kaybı: {val_loss:.4f}, Val Doğruluğu: %{val_acc:5.2f} | LR: {current_lr:.5f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "seed": seed,
            }, best_checkpoint_path)

    total_training_time = time.time() - start_train_time
    print(f"\nEğitim tamamlandı. Toplam Süre: {total_training_time/60:.2f} dakika.")
    print(f"En İyi Val Doğruluğu: %{best_val_acc:.2f} -> Kaydedildi: {best_checkpoint_path}")

    # En iyi model ağırlıklarını yükle ve Bağımsız Test Setinde (10.000 örnek) doğrula
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=-1)
            test_correct += (preds == targets).sum().item()
            test_total += targets.size(0)

    clean_test_acc = (test_correct / test_total) * 100.0
    print("=" * 70)
    print(f"TEMİZ CIFAR-10 TEST KÜMESİ DOĞRULUĞU (CLEAN TEST ACCURACY): %{clean_test_acc:.2f}")
    print("=" * 70)

    history["best_val_acc"] = best_val_acc
    history["clean_test_acc"] = clean_test_acc
    history["training_time_minutes"] = total_training_time / 60.0

    # Log dosyasını kaydet
    log_dir = os.path.join("results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"cifar10_training_curves_seed{seed}.json")  # DÜZELTME (AUDIT A3)
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Eğitim eğrileri kaydedildi -> {log_file}")

    return clean_test_acc, best_val_acc


if __name__ == "__main__":
    # DÜZELTME (AUDIT A3): tohum sabit kodlanmıştı (42); seed 43/44 sürümlenmemiş bir değişiklikle üretilmişti.
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--seed", type=int, default=42)
    _ap.add_argument("--epochs", type=int, default=50)
    _a = _ap.parse_args()
    train_cifar10_resnet18(epochs=_a.epochs, batch_size=128, lr=0.1, seed=_a.seed)

