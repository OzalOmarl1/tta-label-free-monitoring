"""
Veri modülü
===========
* Eğitim için: Parquet CIFAR-10 Dataset (orijinal kod tabanından korunmuştur; aynı
  StratifiedShuffleSplit(test_size=0.10, random_state=seed) ayrımı => her model tohumunun
  kendi 45k/5k ayrımı birebir aynıdır).
* Değerlendirme için: Tüm alanları (temiz CIFAR-10 test, 15 CIFAR-10-C şiddet-5 bozulması,
  CIFAR-10.1 v6, kaynak doğrulama kümesi) uint8 dizileri olarak belleğe alır; normalizasyon
  GPU üzerinde yapılır (torchvision ToTensor+Normalize ile sayısal olarak eşdeğer).
* Akışlar (streams):
    - "iid":    Örneklerin rastgele permütasyonu (stream_seed ile).
    - "sorted": Etikete göre sıralı (sınıf blokları), sınıf sırası ve blok içi sıra rastgele
                => i.i.d. olmayan, zamansal olarak etiket-korelasyonlu akış.
"""

from __future__ import annotations

import io
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset
from torchvision import transforms

from .paths import DATA_DIR

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
CIFAR10_CLASSES = ("airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck")

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
# Değerlendirme alanları: temiz + 15 sentetik + 1 doğal kayma = 17 alan
DOMAINS = ["clean"] + CORRUPTIONS + ["cifar10_1"]


def domain_family(d: str) -> str:
    if d == "clean":
        return "clean"
    if d == "cifar10_1":
        return "natural"
    for fam, lst in CORRUPTION_FAMILIES.items():
        if d in lst:
            return fam
    raise KeyError(d)


# ---------------------------------------------------------------------------
# Eğitim veri kümesi (orijinal kod tabanı)
# ---------------------------------------------------------------------------
class ParquetCIFAR10Dataset(Dataset):
    def __init__(self, parquet_path: str, transform=None):
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        self.targets = self.df["label"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(io.BytesIO(self.df.iloc[idx]["img"]["bytes"])).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, self.targets[idx]


def split_indices(targets: np.ndarray, seed: int, val_ratio: float = 0.10) -> Tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    tr, va = next(splitter.split(np.arange(len(targets)), targets))
    return tr, va


def get_train_val_test_datasets(seed: int):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    test_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)])
    tr_pq = os.path.join(DATA_DIR, "cifar10_train.parquet")
    te_pq = os.path.join(DATA_DIR, "cifar10_test.parquet")
    raw_train = ParquetCIFAR10Dataset(tr_pq, train_tf)
    raw_val = ParquetCIFAR10Dataset(tr_pq, test_tf)
    test = ParquetCIFAR10Dataset(te_pq, test_tf)
    tr_idx, va_idx = split_indices(raw_train.targets, seed)
    return Subset(raw_train, tr_idx), Subset(raw_val, va_idx), test


# ---------------------------------------------------------------------------
# Hızlı değerlendirme dizileri
# ---------------------------------------------------------------------------
def _decode_parquet_uint8(parquet_path: str, cache_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if os.path.exists(cache_path):
        z = np.load(cache_path)
        return z["x"], z["y"]
    df = pd.read_parquet(parquet_path)
    x = np.stack([np.array(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
    y = df["label"].to_numpy(dtype=np.int64)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    np.savez(cache_path, x=x, y=y)
    return x, y


def load_cifar10_test() -> Tuple[np.ndarray, np.ndarray]:
    return _decode_parquet_uint8(os.path.join(DATA_DIR, "cifar10_test.parquet"),
                                 os.path.join(DATA_DIR, "cache", "cifar10_test_uint8.npz"))


def load_cifar10_train() -> Tuple[np.ndarray, np.ndarray]:
    return _decode_parquet_uint8(os.path.join(DATA_DIR, "cifar10_train.parquet"),
                                 os.path.join(DATA_DIR, "cache", "cifar10_train_uint8.npz"))


def load_val_split(seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Model tohumuna ait 5.000'lik temiz kaynak doğrulama kümesi (eğitimde görülmedi)."""
    x, y = load_cifar10_train()
    _, va = split_indices(y, seed)
    return x[va], y[va]


def load_domain(domain: str) -> Tuple[np.ndarray, np.ndarray]:
    if domain == "clean":
        return load_cifar10_test()
    if domain == "cifar10_1":
        x = np.load(os.path.join(DATA_DIR, "cifar10_1", "cifar10.1_v6_data.npy"))
        y = np.load(os.path.join(DATA_DIR, "cifar10_1", "cifar10.1_v6_labels.npy")).astype(np.int64)
        return x, y
    if domain in CORRUPTIONS:
        x = np.load(os.path.join(DATA_DIR, "cifar10_c", f"{domain}_s5.npy"))
        y = np.load(os.path.join(DATA_DIR, "cifar10_c", "labels_s5.npy")).astype(np.int64)
        return x, y
    raise KeyError(domain)


def to_gpu_normalized(x_uint8: np.ndarray, device: str = "cuda") -> torch.Tensor:
    """(N,H,W,3) uint8 -> (N,3,H,W) float32, CIFAR-10 normalizasyonu ile (GPU'da)."""
    t = torch.from_numpy(x_uint8).to(device).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(CIFAR10_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=device).view(1, 3, 1, 1)
    return (t - mean) / std


def make_stream(labels: np.ndarray, kind: str, stream_seed: int) -> np.ndarray:
    """Akış sırası (indeks dizisi). Tüm örnekler tam olarak bir kez kullanılır (tekrar yok)."""
    rng = np.random.default_rng(stream_seed)
    n = len(labels)
    if kind == "iid":
        return rng.permutation(n)
    if kind == "sorted":
        classes = rng.permutation(np.unique(labels))
        parts = [rng.permutation(np.where(labels == c)[0]) for c in classes]
        return np.concatenate(parts)
    raise ValueError(kind)


_CACHE: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}


def cached_domain(domain: str) -> Tuple[np.ndarray, np.ndarray]:
    if domain not in _CACHE:
        _CACHE[domain] = load_domain(domain)
    return _CACHE[domain]
