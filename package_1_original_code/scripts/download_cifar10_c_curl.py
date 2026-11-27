"""
CIFAR-10-C Şiddet 5 Dilimlerini Curl ile Eşzamanlı İndirip .npy Olarak Kaydetme Betiği
"""

import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

HF_BASE_URL = "https://huggingface.co/datasets/YashP54/cifar10-c/resolve/main"
START_BYTE = 122880128
END_BYTE = 153600127
EXPECTED_BYTES = 30720000


def download_corruption_slice(corruption: str, save_dir: str = "data/cifar10_c"):
    os.makedirs(save_dir, exist_ok=True)
    out_npy = os.path.join(save_dir, f"{corruption}_s5.npy")
    if os.path.exists(out_npy):
        try:
            arr = np.load(out_npy)
            if arr.shape == (10000, 32, 32, 3) and arr.dtype == np.uint8:
                print(f"[{corruption}] Zaten mevcut ve doğrulanmış.")
                return out_npy
        except Exception:
            pass

    temp_raw = os.path.join(save_dir, f"{corruption}.raw")
    url = f"{HF_BASE_URL}/{corruption}.npy"
    print(f"[{corruption}] İndiriliyor (30.72 MB)...")

    cmd = [
        "curl.exe", "-L", "-s",
        "-r", f"{START_BYTE}-{END_BYTE}",
        "-o", temp_raw,
        url,
    ]
    t0 = time.time()
    res = subprocess.run(cmd)
    if res.returncode != 0:
        raise RuntimeError(f"curl hatası: {res.returncode}")

    actual_size = os.path.getsize(temp_raw)
    if actual_size != EXPECTED_BYTES:
        raise ValueError(f"Boyut uyuşmazlığı: {actual_size} != {EXPECTED_BYTES}")

    with open(temp_raw, "rb") as f:
        raw_data = f.read()
    arr = np.frombuffer(raw_data, dtype=np.uint8).reshape(10000, 32, 32, 3)
    np.save(out_npy, arr)
    if os.path.exists(temp_raw):
        os.remove(temp_raw)
    print(f"[{corruption}] Başarıyla kaydedildi: {time.time()-t0:.1f} sn -> {out_npy}")
    return out_npy


if __name__ == "__main__":
    t_start = time.time()
    print("CIFAR-10-C (Şiddet 5) Bozulmaları Eşzamanlı İndiriliyor (3 işçi)...")
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(download_corruption_slice, c): c for c in CORRUPTIONS}
        for f in as_completed(futures):
            c = futures[f]
            try:
                f.result()
            except Exception as e:
                print(f"[{c}] İndirme hatası: {e}")
                raise

    print(f"\nTüm bozulmalar tamamlandı. Toplam süre: {(time.time()-t_start)/60:.2f} dakika.")
