"""
Faz 0 Kapsamlı Donanım Kıyaslama Betiği: Source, BN-adapt ve TENT (RTX 4050 GPU)

Ölçülen Değerler:
- Yalnızca GPU çekirdeği değil, Uçtan Uca Süre (End-to-End Time: Veri yükleme + İleri/Geri Yayılım + Metrik Hesabı).
- Parti boyutları: N in {1, 2, 4, 8, 16, 64}.
- Yöntemler: Source, BN-adapt, TENT (N=1 için özdeş BlendedBatchNorm2d fallback).
- 10.000 test örneği üzerinde tam akış.
"""

import os
import sys
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet import get_cifar_resnet18
from src.methods.source import SourceAdapter
from src.methods.bn_adapt import BNAdaptAdapter
from src.methods.tent import TentAdapter, collect_tent_params


def benchmark_end_to_end():
    if not torch.cuda.is_available():
        print("[HATA] CUDA uyumlu GPU bulunamadı!")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    print("=" * 80)
    print("FAZ 0 KAPSAMLI DONANIM VE UÇTAN UCA SÜRE KIYASLAMASI (RTX 4050 GPU)")
    print(f"GPU: {gpu_name} (Toplam VRAM: {total_vram_gb:.2f} GB)")
    print("Ölçüm: Veri Yükleme + Model İleri/Geri Yayılımı + Top-1 Hata / Doğruluk Metriği")
    print("=" * 80)

    total_test_samples = 10000
    batch_sizes = [1, 2, 4, 8, 16, 64]
    methods = ["Source", "BN-adapt", "TENT"]

    # 10.000 test örneği
    synthetic_images = torch.randn(total_test_samples, 3, 32, 32, dtype=torch.float32)
    synthetic_labels = torch.randint(0, 10, (total_test_samples,), dtype=torch.int64)
    dataset = TensorDataset(synthetic_images, synthetic_labels)

    results = {
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "total_samples": total_test_samples,
        "measurements": {},
    }

    for method_name in methods:
        results["measurements"][method_name] = {}
        print(f"\n==================== YÖNTEM: {method_name} ====================")

        for bsz in batch_sizes:
            loader = DataLoader(dataset, batch_size=bsz, shuffle=False)

            # Temiz kaynak modeli başlat
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            base_model = get_cifar_resnet18(num_classes=10).to(device)

            if method_name == "Source":
                adapter = SourceAdapter(base_model)
            elif method_name == "BN-adapt":
                adapter = BNAdaptAdapter(base_model, blend_lambda=0.9)
            elif method_name == "TENT":
                # TENT optimizasyonu
                # DÜZELTME (AUDIT A4): Optimizer BN dönüşümünden ÖNCE model.parameters() ile kurulunca
                # yeni BlendedBatchNorm2d afin parametrelerini içermiyordu => TENT hiç güncellenmiyordu.
                # optimizer=None ile TentAdapter, dönüşümden SONRA doğru parametreleri toplar.
                adapter = TentAdapter(base_model, optimizer=None, lr=1e-3, steps=1, blend_lambda=0.9)

            # Isınma (Warm-up)
            warm_loader = DataLoader(dataset, batch_size=bsz, shuffle=False)
            for i, (imgs, _) in enumerate(warm_loader):
                if i >= 2:
                    break
                _ = adapter.adapt_and_predict(imgs.to(device))

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats(device)

            # UÇTAN UCA SÜRE (End-to-End Time) ÖLÇÜMÜ:
            # Zamanlayıcı veri çekme döngüsünden önce başlar, metrik toplamından sonra biter
            start_wall_time = time.perf_counter()
            correct_count = 0
            total_count = 0

            for imgs, lbls in loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)

                preds = adapter.adapt_and_predict(imgs)
                
                # Metrik hesaplama (Top-1 Accuracy)
                predicted_classes = preds.argmax(dim=-1)
                correct_count += (predicted_classes == lbls).sum().item()
                total_count += lbls.size(0)

            torch.cuda.synchronize()
            elapsed_wall_s = time.perf_counter() - start_wall_time
            peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
            accuracy = correct_count / total_count * 100.0

            throughput = total_test_samples / elapsed_wall_s

            print(
                f"  N = {bsz:2d} | Uçtan Uca: {elapsed_wall_s:6.2f} sn ({elapsed_wall_s/60:4.2f} dk) | "
                f"Hız: {throughput:6.1f} örn/sn | Tepe VRAM: {peak_vram_mb:5.1f} MB"
            )

            results["measurements"][method_name][f"N_{bsz}"] = {
                "batch_size": bsz,
                "end_to_end_seconds": elapsed_wall_s,
                "end_to_end_minutes": elapsed_wall_s / 60.0,
                "throughput_samples_per_sec": throughput,
                "peak_vram_mb": peak_vram_mb,
            }

    out_file = os.path.join("results", "logs", "phase0_e2e_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(f"Uçtan uca kıyaslama tamamlandı. Sonuçlar kaydedildi -> {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    benchmark_end_to_end()

