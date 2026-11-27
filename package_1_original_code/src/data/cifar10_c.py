"""
CIFAR-10-C (Sentetik Bozulmalar) Veri Yükleyici ve İndirici Modülü

Hendrycks & Dietterich (ICLR 2019) tarafından tanımlanan 15 standart bozulma:
- Noise (Gürültü): gaussian_noise, shot_noise, impulse_noise
- Blur (Bulanıklık): defocus_blur, glass_blur, motion_blur, zoom_blur
- Weather (Hava): snow, frost, fog, brightness
- Digital (Dijital): contrast, elastic_transform, pixelate, jpeg_compression

Hugging Face CDN üzerinden HTTP Range istekleriyle yalnızca Şiddet 5 (Severity 5,
son 10.000 örnek, 40000:50000) dilimi indirilerek yerel olarak önbelleğe alınır.
"""

import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from src.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD

CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]

CORRUPTION_FAMILIES = {
    "noise": ["gaussian_noise", "shot_noise", "impulse_noise"],
    "blur": ["defocus_blur", "glass_blur", "motion_blur", "zoom_blur"],
    "weather": ["snow", "frost", "fog", "brightness"],
    "digital": ["contrast", "elastic_transform", "pixelate", "jpeg_compression"],
}

HF_BASE_URL = "https://huggingface.co/datasets/YashP54/cifar10-c/resolve/main"
NPY_HEADER_OFFSET = 128
IMG_BYTES = 3072  # 32 * 32 * 3
SEV5_START = NPY_HEADER_OFFSET + 40000 * IMG_BYTES
SEV5_END = NPY_HEADER_OFFSET + 50000 * IMG_BYTES - 1


def download_cifar10_c_corruption(corruption: str, save_dir: str = "data/cifar10_c") -> str:
    """
    Belirli bir bozulmanın 5. şiddet seviyesini Hugging Face'den indirir ve kaydeder.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{corruption}_s5.npy")
    if os.path.exists(out_path):
        return out_path

    url = f"{HF_BASE_URL}/{corruption}.npy"
    req = urllib.request.Request(url, headers={"Range": f"bytes={SEV5_START}-{SEV5_END}"})

    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[{corruption}] İndiriliyor (Deneme {attempt}/{max_retries})...")
            t0 = time.time()
            resp = urllib.request.urlopen(req, timeout=60)
            raw = resp.read()
            if len(raw) != 10000 * IMG_BYTES:
                raise ValueError(f"Eksik bayt: {len(raw)} != {10000 * IMG_BYTES}")
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(10000, 32, 32, 3)
            np.save(out_path, arr)
            print(f"[{corruption}] Tamamlandı: {time.time()-t0:.1f} sn -> {out_path}")
            return out_path
        except Exception as e:
            print(f"[{corruption}] Hata: {e}")
            if attempt == max_retries:
                raise
            time.sleep(2)


def ensure_cifar10_c_labels(save_dir: str = "data/cifar10_c") -> str:
    """
    CIFAR-10-C etiket dosyasını indirir ve Şiddet 5 etiketlerini kaydeder.
    """
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "labels_s5.npy")
    if os.path.exists(out_path):
        return out_path

    src_label_file = "data/cifar10_c_labels_s5.npy"
    if os.path.exists(src_label_file):
        labels = np.load(src_label_file)
        np.save(out_path, labels)
        return out_path

    url = f"{HF_BASE_URL}/labels.npy"
    print("CIFAR-10-C etiketleri indiriliyor...")
    urllib.request.urlretrieve(url, "data/cifar10_c_labels_full.npy")
    all_labels = np.load("data/cifar10_c_labels_full.npy")
    s5_labels = all_labels[40000:50000]
    np.save(out_path, s5_labels)
    return out_path


def download_all_cifar10_c(save_dir: str = "data/cifar10_c", max_workers: int = 4):
    """
    Tüm 15 bozulmayı eşzamanlı indirir.
    """
    ensure_cifar10_c_labels(save_dir)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_cifar10_c_corruption, c, save_dir): c for c in CORRUPTIONS}
        for f in as_completed(futures):
            c = futures[f]
            try:
                f.result()
            except Exception as e:
                print(f"[{c}] İndirme başarısız: {e}")
                raise


class CIFAR10CDataset(Dataset):
    """
    Tekil bir CIFAR-10-C bozulması (Şiddet 5) için PyTorch Veri Kümesi
    """

    def __init__(self, corruption: str, data_dir: str = "data/cifar10_c", transform=None):
        assert corruption in CORRUPTIONS, f"Bilinmeyen bozulma: {corruption}"
        data_path = os.path.join(data_dir, f"{corruption}_s5.npy")
        label_path = os.path.join(data_dir, "labels_s5.npy")

        if not os.path.exists(data_path):
            download_cifar10_c_corruption(corruption, data_dir)
        if not os.path.exists(label_path):
            ensure_cifar10_c_labels(data_dir)

        self.data = np.load(data_path)
        self.labels = np.load(label_path).astype(np.int64)
        assert len(self.data) == len(self.labels) == 10000

        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def get_cifar10_c_dataloader(
    corruption: str,
    batch_size: int = 64,
    shuffle: bool = False,
    num_workers: int = 2,
    seed: int = 42,
    data_dir: str = "data/cifar10_c",
) -> DataLoader:
    """
    Bozulma için DataLoader üretir.
    """
    dataset = CIFAR10CDataset(corruption, data_dir=data_dir)
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
