"""
Faz 1: CIFAR-10-C (Şiddet 5) Üzerinde Source, BN-adapt ve TENT Değerlendirme Betiği

Protokol:
- Veri Kümesi: CIFAR-10-C, Şiddet 5 (15 bozulma türü, 10.000 test örneği/bozulma)
- Yöntemler:
  1. Source (Adaptasyonsuz dondurulmuş kaynak ResNet-18)
  2. BN-adapt (Test anında anlık parti istatistikleri ile BN güncelleme)
  3. TENT (Entropi minimizasyonu ile BN afin parametreleri güncelleme, lr=1e-3 [VERIFY])
- Parti Boyutu: N = 64
- Tohumlar (Seeds): 42, 43, 44 (Veri akış permütasyon tohumları)
- Episodik Sıfırlama: Her bozulma ve her tohum koğumundan önce model orijinal
  dondurulmuş kontrol noktasına sıfırlanır.
- Bütünlük Denetimi: SHA256 hash doğrulaması, gradyan izolasyonu, etiket sızıntısı kontrolü.
"""

import os
import sys
import time
import json
import hashlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.models.resnet import get_cifar_resnet18
from src.methods.source import SourceAdapter
from src.methods.bn_adapt import BNAdaptAdapter
from src.methods.tent import TentAdapter
from src.data.cifar10_c import CORRUPTIONS, CORRUPTION_FAMILIES, get_cifar10_c_dataloader


def compute_ece(probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15) -> float:
    """
    Beklenen Kalibrasyon Hatası (Expected Calibration Error - ECE) hesaplar.
    """
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = predictions.eq(targets)

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]

        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.float().mean().item()

        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean().item()
            avg_confidence_in_bin = confidences[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    return ece * 100.0


def verify_checkpoint_integrity(checkpoint_path: str, hash_log_path: str = "results/logs/checkpoint_hashes.json",
                                seed: int = None) -> str:
    """
    Kontrol noktasının SHA256 hash değerini doğrular.
    """
    assert os.path.exists(checkpoint_path), f"Checkpoint bulunamadı: {checkpoint_path}"
    assert os.path.exists(hash_log_path), f"Hash kayıt dosyası bulunamadı: {hash_log_path}"

    with open(checkpoint_path, "rb") as f:
        actual_hash = hashlib.sha256(f.read()).hexdigest()

    with open(hash_log_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    # DÜZELTME (AUDIT A2): Hash dosyası tohum bazlı ({'seed42': {...}}) formata geçtiği için eski
    # meta['sha256'] erişimi KeyError veriyordu (run_phase1.py çalışmıyordu).
    if "sha256" in meta:
        expected_hash = meta["sha256"]
    else:
        if seed is None:
            import re as _re
            m = _re.search(r"seed(\d+)", os.path.basename(checkpoint_path))
            seed = int(m.group(1)) if m else None
        expected_hash = meta[f"seed{seed}"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError(
            f"KRİTİK HATA: Checkpoint SHA256 uyuşmazlığı!\n"
            f"Beklenen: {expected_hash}\n"
            f"Hesaplanan: {actual_hash}"
        )
    return actual_hash


def load_clean_model(checkpoint_path: str, device: str = "cuda") -> nn.Module:
    """
    Temiz kaynak modeli kontrol noktasından yükler.
    """
    model = get_cifar_resnet18(num_classes=10)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    return model


def evaluate_single_run(
    method_name: str,
    corruption: str,
    seed: int,
    checkpoint_path: str,
    batch_size: int = 64,
    device: str = "cuda",
) -> dict:
    """
    Tek bir (Yöntem, Bozulma, Tohum) koğumu için TTA değerlendirmesi yapar.
    """
    # 1. Her koğumda sıfırdan temiz modeli yükle (Episodik sıfırlama)
    model = load_clean_model(checkpoint_path, device=device)

    # 2. Yöntem yöneticisini oluştur
    if method_name == "Source":
        adapter = SourceAdapter(model)
    elif method_name == "BN-adapt":
        adapter = BNAdaptAdapter(model, blend_lambda=0.9)
    elif method_name == "TENT":
        # NOT (AUDIT A5): Faz-1 sonuçları 'güncelleme-sonrası tahmin' ile üretildi; yeniden üretilebilirlik için
        # açıkça korunuyor. Resmi Tent protokolü predict='official' (bkz. Final/ttamon/methods.py).
        adapter = TentAdapter(model, lr=1e-3, steps=1, blend_lambda=0.9, predict="after_update")
    else:
        raise ValueError(f"Bilinmeyen yöntem: {method_name}")

    # 3. Belirtilen tohum ile permüte edilmiş veri yükleyicisini al
    loader = get_cifar10_c_dataloader(
        corruption=corruption,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        seed=seed,
    )

    all_preds = []
    all_targets = []
    all_probs = []

    start_time = time.time()

    for imgs, targets in loader:
        imgs, targets = imgs.to(device), targets.to(device)
        logits = adapter.adapt_and_predict(imgs)
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)

        all_preds.append(preds.detach().cpu())
        all_targets.append(targets.detach().cpu())
        all_probs.append(probs.detach().cpu())

    elapsed_time = time.time() - start_time

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    all_probs = torch.cat(all_probs)

    correct = (all_preds == all_targets).sum().item()
    total = len(all_targets)
    accuracy = (correct / total) * 100.0
    ece = compute_ece(all_probs, all_targets)

    # Anomali / Sızıntı kontrolü
    is_suspicious = False
    suspicion_note = ""
    if accuracy > 92.0 and corruption in ["gaussian_noise", "shot_noise", "impulse_noise"]:
        is_suspicious = True
        suspicion_note = f"Şiddet 5 gürültüde aşırı yüksek doğruluk (%{accuracy:.2f}) - sızıntı denetimi gerekli."

    return {
        "method": method_name,
        "corruption": corruption,
        "seed": seed,
        "accuracy": round(accuracy, 2),
        "ece": round(ece, 2),
        "elapsed_seconds": round(elapsed_time, 2),
        "samples": total,
        "is_suspicious": is_suspicious,
        "suspicion_note": suspicion_note,
    }


def compute_bootstrap_ci(data: list, n_boot: int = 1000, ci: float = 0.95, seed: int = 42) -> tuple:
    """
    Yeniden örnekleme (bootstrap) ile güven aralığı hesaplar.
    """
    if len(data) == 0:
        return 0.0, 0.0
    rng = np.random.RandomState(seed)
    n = len(data)
    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(data, size=n, replace=True)
        boot_means.append(np.mean(sample))
    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return round(lower, 2), round(upper, 2)
