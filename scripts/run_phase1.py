"""
Faz 1 Deney Yürütücüsü:
CIFAR-10-C (Şiddet 5) Üzerinde Source, BN-adapt ve TENT (N=64, 3 Tohum)

Değerlendirme Protokolü:
- Model: ResNet-18 (Dondurulmuş Checkpoint: src/models/checkpoints/resnet18_cifar10_seed42.pth)
- Bütünlük Kontrolü: results/logs/checkpoint_hashes.json üzerinden SHA256 teyidi.
- Yöntemler:
  * Source (Baseline / Dondurulmuş çıkarım)
  * BN-adapt (Anlık parti istatistikleri ile test-time BN)
  * TENT (Entropi minimizasyonu, lr=1e-3 [VERIFY], 1 adım)
- Parti Boyutu: N=64
- Tohumlar (3 seeds): 42, 43, 44 (Veri akışı permütasyon tohumları)
- Bozulmalar (15 standart CIFAR-10-C bozulması):
  * Noise: gaussian_noise, shot_noise, impulse_noise
  * Blur: defocus_blur, glass_blur, motion_blur, zoom_blur
  * Weather: snow, frost, fog, brightness
  * Digital: contrast, elastic_transform, pixelate, jpeg_compression
- Çıktılar:
  * results/logs/phase1_cifar10_c_results.json (Ham ve özet sonuçlar)
  * Bozulma bazlı ortalama doğruluk ve %95 Bootstrap Güven Aralıkları (CIs).
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import time
import json
import numpy as np
import torch

from src.evaluate_phase1 import (
    verify_checkpoint_integrity,
    evaluate_single_run,
    compute_bootstrap_ci,
)
from src.data.cifar10_c import CORRUPTIONS, CORRUPTION_FAMILIES

CHECKPOINT_PATH = "src/models/checkpoints/resnet18_cifar10_seed42.pth"
HASH_LOG_PATH = "results/logs/checkpoint_hashes.json"
RESULTS_PATH = "results/logs/phase1_cifar10_c_results.json"
SEEDS = [42, 43, 44]
METHODS = ["Source", "BN-adapt", "TENT"]
BATCH_SIZE = 64


def run_phase1_experiments():
    print("=" * 80)
    print("FAZ 1: CIFAR-10-C (ŞİDDET 5) TTA DENEYLERİ BAŞLATILIYOR")
    print(f"Yöntemler: {METHODS} | Batch Size: {BATCH_SIZE} | Tohumlar: {SEEDS}")
    print("=" * 80)

    # 1. Checkpoint SHA256 bütünlüğünü doğrula
    print("\n[Adım 1/3] Kontrol noktası SHA256 hash bütünlüğü doğrulanıyor...")
    ckpt_hash = verify_checkpoint_integrity(CHECKPOINT_PATH, HASH_LOG_PATH)
    print(f"-> Doğrulandı! SHA256: {ckpt_hash}")

    # 2. Deney matrisini yürüt
    print("\n[Adım 2/3] Deneyler yürütülüyor (15 bozulma x 3 yöntem x 3 tohum = 135 koğum)...")
    results = []
    total_runs = len(CORRUPTIONS) * len(METHODS) * len(SEEDS)
    current_run = 0
    t_start_total = time.time()

    for corruption in CORRUPTIONS:
        print(f"\n--- Bozulma: {corruption.upper()} (Şiddet 5) ---")
        for method in METHODS:
            for seed in SEEDS:
                current_run += 1
                t0 = time.time()
                res = evaluate_single_run(
                    method_name=method,
                    corruption=corruption,
                    seed=seed,
                    checkpoint_path=CHECKPOINT_PATH,
                    batch_size=BATCH_SIZE,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                dt = time.time() - t0
                results.append(res)
                print(
                    f"[{current_run:3d}/{total_runs:3d}] {method:10s} | Seed {seed} | "
                    f"Doğruluk: %{res['accuracy']:5.2f} | ECE: %{res['ece']:5.2f} | "
                    f"Süre: {dt:.2f}s"
                )

    total_duration = time.time() - t_start_total
    print(f"\nTüm koğumlar tamamlandı! Toplam süre: {total_duration/60:.2f} dakika.")

    # 3. İstatistiksel Analiz: Ortalama, İki Aşamalı Kümeleme Bootstrap CIs ve Aile Özetleri
    print("\n[Adım 3/3] İstatistiksel analiz ve bootstrap güven aralıkları hesaplanıyor...")

    summary_by_corruption = {}
    for c in CORRUPTIONS:
        summary_by_corruption[c] = {}
        for m in METHODS:
            runs = [r for r in results if r["corruption"] == c and r["method"] == m]
            accs = [r["accuracy"] for r in runs]
            eces = [r["ece"] for r in runs]
            mean_acc = round(float(np.mean(accs)), 2)
            std_acc = round(float(np.std(accs)), 2)
            mean_ece = round(float(np.mean(eces)), 2)
            ci_lower, ci_upper = compute_bootstrap_ci(accs, n_boot=1000, ci=0.95, seed=42)

            summary_by_corruption[c][m] = {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "ci_95_acc": [ci_lower, ci_upper],
                "mean_ece": mean_ece,
                "seed_accs": accs,
            }

    # Aile bazlı ortalamalar
    family_summary = {}
    for fam_name, fam_corruptions in CORRUPTION_FAMILIES.items():
        family_summary[fam_name] = {}
        for m in METHODS:
            fam_accs = [
                summary_by_corruption[c][m]["mean_acc"]
                for c in fam_corruptions
            ]
            family_summary[fam_name][m] = round(float(np.mean(fam_accs)), 2)

    # Genel ortalama (15 bozulma)
    overall_summary = {}
    for m in METHODS:
        all_accs = [summary_by_corruption[c][m]["mean_acc"] for c in CORRUPTIONS]
        all_eces = [summary_by_corruption[c][m]["mean_ece"] for c in CORRUPTIONS]
        overall_summary[m] = {
            "mean_acc": round(float(np.mean(all_accs)), 2),
            "mean_ece": round(float(np.mean(all_eces)), 2),
            "ci_95_indicative": list(compute_bootstrap_ci(all_accs, n_boot=1000, ci=0.95, seed=42)),
        }

    output_payload = {
        "metadata": {
            "phase": "Phase 1",
            "model": "ResNet-18 (CIFAR-10)",
            "checkpoint_sha256": ckpt_hash,
            "dataset": "CIFAR-10-C (Severity 5)",
            "batch_size": BATCH_SIZE,
            "seeds": SEEDS,
            "total_runs": total_runs,
            "total_duration_minutes": round(total_duration / 60.0, 2),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "overall_summary": overall_summary,
        "family_summary": family_summary,
        "summary_by_corruption": summary_by_corruption,
        "raw_runs": results,
    }

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)

    print(f"Tüm Faz 1 sonuçları kaydedildi -> {RESULTS_PATH}")
    return output_payload


if __name__ == "__main__":
    # KULLANIMDAN KALDIRILDI (AUDIT A2/A8): Bu betik tek checkpoint (seed42) ile 3 'tohum' koşturuyordu
    # (model varyansı yok) ve sonuçları revize betikle AYNI dosyanın üzerine yazıyordu.
    raise SystemExit('run_phase1.py kullanımdan kaldırıldı; scripts/run_phase1_revised.py kullanın.')
