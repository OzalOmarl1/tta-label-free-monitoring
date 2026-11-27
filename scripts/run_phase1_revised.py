"""
Faz 1 Revize Deney Yürütücüsü:
CIFAR-10-C (Şiddet 5) Üzerinde Source, BN-adapt ve TENT (N=64, 3 Bağımsız Tohum)

METODOLOJİK DÜZELTME (Tasarım Kusuru Giderildi):
- Her tohum (Seed 42, 43, 44) bağımsız olarak eğitilmiş kendi kaynak modelini
  (resnet18_cifar10_seed{seed}.pth) ve kendi test akış sırasını kullanır.
- Bütünlük Kontrolü: 3 modelin SHA256 özetleri checkpoint_hashes.json üzerinden doğrulanır.
- İstatistiksel Analiz:
  * Eşleştirilmiş farklar: Delta = TENT - BN-adapt
  * İki aşamalı kümeleme bootstrap (two-stage cluster bootstrap) ile %95 güven aralıkları
    (Aşama 1: Tohumlar, Aşama 2: Tohum içi bozulmalar/örnekler).
  * 3 tohumla güven aralıklarının yaklaşık ve yalnızca gösterge niteliğinde olduğu notu.
  * Contrast ve JPEG anomalilerinde nedensel yorum yapılmaz (nedeni bilinmeyen gözlem).
"""

import os
import sys

sys.path.insert(0, os.path.abspath("."))

import time
import json
import hashlib
import numpy as np
import torch

from src.evaluate_phase1 import evaluate_single_run, compute_bootstrap_ci
from src.data.cifar10_c import CORRUPTIONS, CORRUPTION_FAMILIES

CHECKPOINTS = {
    42: "src/models/checkpoints/resnet18_cifar10_seed42.pth",
    43: "src/models/checkpoints/resnet18_cifar10_seed43.pth",
    44: "src/models/checkpoints/resnet18_cifar10_seed44.pth",
}
HASH_LOG_PATH = "results/logs/checkpoint_hashes.json"
RESULTS_PATH = "results/logs/phase1_cifar10_c_results.json"
SEEDS = [42, 43, 44]
METHODS = ["Source", "BN-adapt", "TENT"]
BATCH_SIZE = 64


def verify_all_checkpoints():
    assert os.path.exists(HASH_LOG_PATH), "Hash kayıt dosyası bulunamadı!"
    with open(HASH_LOG_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)

    for seed, ckpt_path in CHECKPOINTS.items():
        assert os.path.exists(ckpt_path), f"Checkpoint bulunamadı: {ckpt_path}"
        actual_hash = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
        expected_hash = meta[f"seed{seed}"]["sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"Seed {seed} hash uyuşmazlığı! Beklenen: {expected_hash}, Bulunan: {actual_hash}")
        print(f"-> Seed {seed} Checkpoint Doğrulandı: {actual_hash[:16]}... (Clean Acc: %{meta[f'seed{seed}']['clean_test_acc']:.2f})")


def two_stage_cluster_bootstrap_paired(
    tent_matrix: np.ndarray,
    bn_matrix: np.ndarray,
    n_boot: int = 2000,
    ci: float = 0.95,
    random_seed: int = 42,
) -> tuple:
    """
    Eşleştirilmiş farklar (Delta = TENT - BN-adapt) için iki aşamalı kümeleme bootstrap.
    tent_matrix: (n_seeds, n_corruptions)
    bn_matrix: (n_seeds, n_corruptions)
    """
    rng = np.random.RandomState(random_seed)
    diff_matrix = tent_matrix - bn_matrix  # (n_seeds, n_corruptions)
    n_seeds, n_corruptions = diff_matrix.shape

    boot_means = []
    for _ in range(n_boot):
        # Aşama 1: Tohumları yeniden örnekle (kümeler)
        seed_idx = rng.choice(n_seeds, size=n_seeds, replace=True)
        sampled_cluster = diff_matrix[seed_idx, :]  # (n_seeds, n_corruptions)

        # Aşama 2: Küme içi bozulmaları yeniden örnekle
        corrupt_idx = rng.choice(n_corruptions, size=n_corruptions, replace=True)
        boot_sample = sampled_cluster[:, corrupt_idx]

        boot_means.append(np.mean(boot_sample))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100.0))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100.0))
    return round(lower, 2), round(upper, 2)


def run_phase1_independent_seeds():
    print("=" * 80)
    print("FAZ 1 REVIZE: CIFAR-10-C (ŞİDDET 5) BAĞIMSIZ MODEL TOHUMLARIYLA TTA")
    print(f"Tohumlar: {SEEDS} (Her tohum bağımsız eğitilmiş kendi modelini kullanır)")
    print("=" * 80)

    # 1. Bütünlük doğrulaması
    print("\n[Adım 1/3] 3 Bağımsız Kontrol Noktasının SHA256 Hash'leri Doğrulanıyor...")
    verify_all_checkpoints()

    # 2. Deney koğumları (15 bozulma x 3 yöntem x 3 bağımsız tohum = 135 koğum)
    print("\n[Adım 2/3] Deneyler yürütülüyor...")
    results = []
    total_runs = len(CORRUPTIONS) * len(METHODS) * len(SEEDS)
    current_run = 0
    t_start_total = time.time()

    for corruption in CORRUPTIONS:
        print(f"\n--- Bozulma: {corruption.upper()} (Şiddet 5) ---")
        for method in METHODS:
            for seed in SEEDS:
                current_run += 1
                ckpt_path = CHECKPOINTS[seed]
                t0 = time.time()
                res = evaluate_single_run(
                    method_name=method,
                    corruption=corruption,
                    seed=seed,
                    checkpoint_path=ckpt_path,
                    batch_size=BATCH_SIZE,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )
                dt = time.time() - t0
                results.append(res)
                print(
                    f"[{current_run:3d}/{total_runs:3d}] {method:10s} | Seed {seed} (Model+Stream) | "
                    f"Doğruluk: %{res['accuracy']:5.2f} | ECE: %{res['ece']:5.2f} | Süre: {dt:.2f}s"
                )

    total_duration = time.time() - t_start_total
    print(f"\nTüm koğumlar tamamlandı! Toplam süre: {total_duration/60:.2f} dakika.")

    # 3. İstatistiksel Analiz: Eşleştirilmiş Farklar ve Kümeleme Bootstrap CIs
    print("\n[Adım 3/3] İstatistiksel analiz ve eşleştirilmiş farklar hesaplanıyor...")

    summary_by_corruption = {}
    n_corruptions = len(CORRUPTIONS)
    n_seeds = len(SEEDS)

    tent_grid = np.zeros((n_seeds, n_corruptions))
    bn_grid = np.zeros((n_seeds, n_corruptions))
    src_grid = np.zeros((n_seeds, n_corruptions))

    for c_idx, c in enumerate(CORRUPTIONS):
        summary_by_corruption[c] = {}
        for m in METHODS:
            runs = [r for r in results if r["corruption"] == c and r["method"] == m]
            accs = [r["accuracy"] for r in runs]
            eces = [r["ece"] for r in runs]

            if m == "Source":
                src_grid[:, c_idx] = accs
            elif m == "BN-adapt":
                bn_grid[:, c_idx] = accs
            elif m == "TENT":
                tent_grid[:, c_idx] = accs

            mean_acc = round(float(np.mean(accs)), 2)
            std_acc = round(float(np.std(accs)), 2)
            mean_ece = round(float(np.mean(eces)), 2)
            ci_lower, ci_upper = compute_bootstrap_ci(accs, n_boot=2000, ci=0.95, seed=42)

            summary_by_corruption[c][m] = {
                "mean_acc": mean_acc,
                "std_acc": std_acc,
                "ci_95_acc": [ci_lower, ci_upper],
                "mean_ece": mean_ece,
                "seed_accs": accs,
            }

        # Eşleştirilmiş fark: Delta = TENT - BN-adapt (her tohum için kendi modeliyle)
        c_deltas = tent_grid[:, c_idx] - bn_grid[:, c_idx]
        delta_mean = round(float(np.mean(c_deltas)), 2)
        delta_std = round(float(np.std(c_deltas)), 2)
        delta_ci_l, delta_ci_u = compute_bootstrap_ci(list(c_deltas), n_boot=2000, ci=0.95, seed=42)
        summary_by_corruption[c]["paired_delta_tent_minus_bn"] = {
            "mean_delta": delta_mean,
            "std_delta": delta_std,
            "ci_95_delta": [delta_ci_l, delta_ci_u],
            "seed_deltas": [round(float(d), 2) for d in c_deltas],
        }

    # İki Aşamalı Kümeleme Bootstrap: Genel Eşleştirilmiş Fark
    overall_delta_ci = two_stage_cluster_bootstrap_paired(tent_grid, bn_grid, n_boot=2000, ci=0.95, random_seed=42)
    overall_mean_delta = round(float(np.mean(tent_grid - bn_grid)), 2)

    # Aile bazlı özetler
    family_summary = {}
    for fam_name, fam_corruptions in CORRUPTION_FAMILIES.items():
        family_summary[fam_name] = {}
        fam_indices = [CORRUPTIONS.index(c) for c in fam_corruptions]
        for m in METHODS:
            fam_accs = [summary_by_corruption[c][m]["mean_acc"] for c in fam_corruptions]
            family_summary[fam_name][m] = round(float(np.mean(fam_accs)), 2)
        fam_delta = np.mean(tent_grid[:, fam_indices] - bn_grid[:, fam_indices])
        family_summary[fam_name]["delta_tent_minus_bn"] = round(float(fam_delta), 2)

    # Genel ortalama
    overall_summary = {}
    for m in METHODS:
        all_accs = [summary_by_corruption[c][m]["mean_acc"] for c in CORRUPTIONS]
        all_eces = [summary_by_corruption[c][m]["mean_ece"] for c in CORRUPTIONS]
        overall_summary[m] = {
            "mean_acc": round(float(np.mean(all_accs)), 2),
            "mean_ece": round(float(np.mean(all_eces)), 2),
            "ci_95_indicative": list(compute_bootstrap_ci(all_accs, n_boot=2000, ci=0.95, seed=42)),
        }

    overall_summary["paired_delta_tent_minus_bn"] = {
        "mean_delta": overall_mean_delta,
        "two_stage_cluster_bootstrap_ci_95": list(overall_delta_ci),
        "ci_caveat": "3 tohum kullanıldığından güven aralıkları yaklaşık ve yalnızca gösterge niteliğindedir (indicative only).",
    }

    output_payload = {
        "metadata": {
            "phase": "Phase 1 Revised",
            "protocol": "Independent source training seeds + test stream seeds",
            "model_seeds": {
                "42": {"ckpt": CHECKPOINTS[42], "clean_acc": 93.86},
                "43": {"ckpt": CHECKPOINTS[43], "clean_acc": 93.44},
                "44": {"ckpt": CHECKPOINTS[44], "clean_acc": 93.88},
            },
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

    print(f"\nTüm revize Faz 1 sonuçları kaydedildi -> {RESULTS_PATH}")
    return output_payload


if __name__ == "__main__":
    run_phase1_independent_seeds()

