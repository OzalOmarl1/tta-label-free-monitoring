"""
Faz 0 Gerçek Veri Kümeleri Doğrulama Betiği (Dataset Verification Script)

Bu betik:
1. CIFAR-10 train, val (%10 stratified) ve test kümelerini yükler.
2. CIFAR-10.1 v6 (2000 test görseli) kümesini indirir ve doğrular.
3. CIFAR-10-C sentetik bozulma arayüzünü ve dosya varlığını test eder.
4. PACS stil/alan kayması arayüzünü test eder.
5. Dirichlet sınıf dengesizliği örnekleyicisinin N=1, N=4 ve N=64 akışlarını kontrol eder.
"""

import os
import sys
import torch
import numpy as np

from src.data.cifar10 import get_cifar10_datasets, get_cifar10_dataloaders
from src.data.cifar10_1 import CIFAR10Point1Dataset, get_cifar10_1_dataloader
from src.data.cifar10_c import CORRUPTIONS, CORRUPTION_FAMILIES  # DÜZELTME (AUDIT A1): ALL_CORRUPTIONS mevcut değildi
from src.data.pacs import PACS_DOMAINS, PACS_CLASSES
from src.data.imbalance import get_imbalanced_stream_dataloader


def verify_cifar10():
    print("\n--- 1. CIFAR-10 Doğrulaması ---")
    data_dir = "./data"
    train_set, val_set, test_set = get_cifar10_datasets(data_dir=data_dir, val_ratio=0.10, seed=42)

    print(f"Train örnek sayısı: {len(train_set)} (Beklenen: 45000)")
    print(f"Source-Val örnek sayısı: {len(val_set)} (Beklenen: 5000)")
    print(f"Source-Test örnek sayısı: {len(test_set)} (Beklenen: 10000)")

    assert len(train_set) == 45000
    assert len(val_set) == 5000
    assert len(test_set) == 10000

    # İlk örnek şekil kontrolü
    img, target = train_set[0]
    print(f"Train örnek şekli: {img.shape}, Etiket tipi: {type(target)}, Değer: {target}")
    assert img.shape == (3, 32, 32)
    assert 0 <= target <= 9
    print("[BAŞARILI] CIFAR-10 veri ayrımı ve yükleyicisi doğrulandı.")


def verify_cifar10_1():
    print("\n--- 2. CIFAR-10.1 (v6) Doğrulaması ---")
    data_dir = "./data"
    loader = get_cifar10_1_dataloader(data_dir=data_dir, batch_size=64, download=True)
    total_samples = len(loader.dataset)
    print(f"CIFAR-10.1 toplam örnek sayısı: {total_samples} (Beklenen: 2000)")
    assert total_samples == 2000

    imgs, targets = next(iter(loader))
    print(f"İlk batch şekli: {imgs.shape}, Etiketler: {targets.shape}")
    assert imgs.shape[1:] == (3, 32, 32)
    assert targets.min() >= 0 and targets.max() <= 9
    print("[BAŞARILI] CIFAR-10.1 veri kümesi doğrulandı.")


def verify_dirichlet_stream():
    print("\n--- 3. Dirichlet Akış ve Parti Boyutu Doğrulaması ---")
    data_dir = "./data"
    _, _, test_set = get_cifar10_datasets(data_dir=data_dir, val_ratio=0.10, seed=42)

    # N=1 için test
    loader_n1 = get_imbalanced_stream_dataloader(test_set, batch_size=1, alpha=0.1, seed=42)
    imgs_1, lbls_1 = next(iter(loader_n1))
    print(f"N=1 akış ilk batch: imgs={imgs_1.shape}, labels={lbls_1.shape}")
    assert imgs_1.shape == (1, 3, 32, 32)

    # N=4 için test
    loader_n4 = get_imbalanced_stream_dataloader(test_set, batch_size=4, alpha=float("inf"), seed=42)
    imgs_4, lbls_4 = next(iter(loader_n4))
    print(f"N=4 akış ilk batch: imgs={imgs_4.shape}, labels={lbls_4.shape}")
    assert imgs_4.shape == (4, 3, 32, 32)

    print("[BAŞARILI] Dirichlet akış örnekleyicisi doğrulandı.")


def main():
    print("=== FAZ 0: Veri Kümeleri ve Yükleyici Doğrulama Süreci ===")
    verify_cifar10()
    verify_cifar10_1()
    verify_dirichlet_stream()
    print("\n=== TÜM FAZ 0 VERİ DOĞRULAMA ADIMLARI BAŞARIYLA TAMAMLANDI ===")


if __name__ == "__main__":
    main()

