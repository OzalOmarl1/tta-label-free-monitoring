"""
Faz 0 Donanım Kıyaslama Betiği: ResNet-18 TENT Koşumu (RTX 4050 GPU)

Bu betik:
1. Yerel NVIDIA GeForce RTX 4050 Laptop GPU üzerinde ResNet-18 modelini başlatır.
2. Wang et al. (2021) TENT algoritmasını (BN afin optimizasyonu, lr=1e-3 [VERIFY], 1 adım) kurar.
3. 10.000 test örneği (standart CIFAR-10 test boyutu) üzerinden:
   - Farklı parti boyutları (N in {1, 4, 16, 64, 200}) için
   - CUDA zaman ölçümü (torch.cuda.Event ile hassas milisaniye ölçümü)
   - Tepe VRAM tüketimi (torch.cuda.max_memory_allocated)
   ölçümlerini yapar.
4. Ölçülen gerçek değerleri JSON ve konsol tablosu olarak raporlar.
"""

import os
import sys
import time
import json
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.models.resnet import get_cifar_resnet18
from src.methods.tent import configure_tent_model, collect_tent_params, TentAdapter


def benchmark_tent_on_gpu():
    if not torch.cuda.is_available():
        print("[HATA] CUDA uyumlu GPU bulunamadı!")
        sys.exit(1)

    device = torch.device("cuda:0")
    gpu_name = torch.cuda.get_device_name(device)
    total_vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)

    print("=" * 70)
    print("FAZ 0 DONANIM KIYASLAMA TESTİ: RESNET-18 TENT ADAPTASYON KOŞUMU")
    print(f"Kullanılan Donanım: {gpu_name}")
    print(f"Toplam VRAM: {total_vram_gb:.2f} GB")
    print(f"PyTorch Sürümü: {torch.__version__}, CUDA: {torch.version.cuda}")
    print("=" * 70)

    # 10.000 test örneği (CIFAR boyutu: 32x32x3)
    total_test_samples = 10000
    batch_sizes = [1, 4, 16, 64, 200]

    # Bellekte 10.000 yapay örnek oluştur (saf hesaplama ve bellek ölçümü için)
    synthetic_images = torch.randn(total_test_samples, 3, 32, 32, dtype=torch.float32)
    synthetic_labels = torch.randint(0, 10, (total_test_samples,), dtype=torch.int64)
    dataset = TensorDataset(synthetic_images, synthetic_labels)

    benchmark_results = {
        "gpu_name": gpu_name,
        "total_vram_gb": total_vram_gb,
        "total_samples": total_test_samples,
        "configurations": {},
    }

    for bsz in batch_sizes:
        print(f"\n>>> Test Ediliyor: Batch Size N = {bsz} (Toplam {total_test_samples} örnek)...")
        loader = DataLoader(dataset, batch_size=bsz, shuffle=False)

        # Modeli ve TENT uyarlayıcısını sıfırla
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        model = get_cifar_resnet18(num_classes=10).to(device)
        model = configure_tent_model(model)
        params, names = collect_tent_params(model)
        optimizer = optim.Adam(params, lr=1e-3)
        adapter = TentAdapter(model, optimizer, steps=1)

        # Isınma (Warm-up): 3 batch
        warmup_loader = DataLoader(dataset, batch_size=bsz, shuffle=False)
        for i, (imgs, _) in enumerate(warmup_loader):
            if i >= 3:
                break
            imgs = imgs.to(device)
            _ = adapter.adapt_and_predict(imgs)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

        # Hassas CUDA zamanlayıcıları
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        start_wall_time = time.time()

        for imgs, _ in loader:
            imgs = imgs.to(device)
            _ = adapter.adapt_and_predict(imgs)

        end_event.record()
        torch.cuda.synchronize()

        elapsed_cuda_ms = start_event.elapsed_time(end_event)
        elapsed_cuda_s = elapsed_cuda_ms / 1000.0
        elapsed_wall_s = time.time() - start_wall_time
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        samples_per_sec = total_test_samples / elapsed_cuda_s

        print(f"  -> CUDA Süresi: {elapsed_cuda_s:.2f} saniye ({elapsed_cuda_s/60:.2f} dakika)")
        print(f"  -> İşleme Hızı: {samples_per_sec:.1f} örnek/saniye")
        print(f"  -> Tepe VRAM Kullanımı: {peak_vram_mb:.1f} MB (Toplamın %{peak_vram_mb / (total_vram_gb * 1024) * 100:.1f}'i)")

        benchmark_results["configurations"][f"batch_size_{bsz}"] = {
            "batch_size": bsz,
            "elapsed_seconds": elapsed_cuda_s,
            "elapsed_minutes": elapsed_cuda_s / 60.0,
            "samples_per_sec": samples_per_sec,
            "peak_vram_mb": peak_vram_mb,
            "vram_utilization_pct": peak_vram_mb / (total_vram_gb * 1024) * 100.0,
        }

    # Sonuçları JSON olarak kaydet
    out_dir = os.path.join("results", "logs")
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, "phase0_gpu_benchmark.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"Kıyaslama tamamlandı. Sonuçlar kaydedildi -> {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    benchmark_tent_on_gpu()

