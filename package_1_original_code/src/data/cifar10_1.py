"""
CIFAR-10.1 Doğal Dağılım Kayması (Natural Distribution Shift) Veri Kümesi Modülü

CIFAR-10.1 (Recht et al., 2018 / 2019):
CIFAR-10 test setinin toplandığı Tiny Images arama motoru protokolü birebir tekrarlanarak
oluşturulmuş bağımsız ve yeni bir test kümesidir.
v6 sürümü toplam 2.000 test görseli ve etiketini içerir.

Bu modül:
1. CIFAR-10.1 v6 verisini resmi GitHub deposundan indirir (eğer yoksa).
2. Veriyi NumPy formatından PyTorch Dataset'e dönüştürür.
3. CIFAR-10 ile aynı normalizasyonu uygular.
"""

import os
import urllib.request
from typing import Optional, Callable
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from src.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD

CIFAR10_1_URL_DATA = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_data.npy"
CIFAR10_1_URL_LABELS = "https://raw.githubusercontent.com/modestyachts/CIFAR-10.1/master/datasets/cifar10.1_v6_labels.npy"


class CIFAR10Point1Dataset(Dataset):
    """
    CIFAR-10.1 v6 Test Veri Kümesi PyTorch Dataset sınıfı.
    """

    def __init__(
        self,
        root: str = "./data",
        transform: Optional[Callable] = None,
        download: bool = True,
    ):
        """
        Args:
            root: Dosyaların saklanacağı temel dizin.
            transform: Görsel dönüşümleri.
            download: Veri yoksa otomatik indirilsin mi?
        """
        self.root = os.path.join(root, "cifar10_1")
        self.transform = transform
        os.makedirs(self.root, exist_ok=True)

        data_path = os.path.join(self.root, "cifar10.1_v6_data.npy")
        labels_path = os.path.join(self.root, "cifar10.1_v6_labels.npy")

        if download and (not os.path.exists(data_path) or not os.path.exists(labels_path)):
            self._download(data_path, labels_path)

        if not os.path.exists(data_path) or not os.path.exists(labels_path):
            raise RuntimeError(
                f"CIFAR-10.1 verisi {self.root} içinde bulunamadı. "
                "Lütfen download=True olarak ayarlayın."
            )

        self.data = np.load(data_path)  # (2000, 32, 32, 3) uint8
        self.targets = np.load(labels_path).astype(np.int64)  # (2000,) int64

        assert len(self.data) == len(self.targets) == 2000, (
            f"CIFAR-10.1 v6 beklenen 2000 örnek yerine {len(self.data)} örnek içeriyor!"
        )

    def _download(self, data_path: str, labels_path: str) -> None:
        """Veri ve etiket dosyalarını resmi GitHub deposundan indirir."""
        if not os.path.exists(data_path):
            print(f"CIFAR-10.1 v6 görselleri indiriliyor -> {data_path}...")
            urllib.request.urlretrieve(CIFAR10_1_URL_DATA, data_path)

        if not os.path.exists(labels_path):
            print(f"CIFAR-10.1 v6 etiketleri indiriliyor -> {labels_path}...")
            urllib.request.urlretrieve(CIFAR10_1_URL_LABELS, labels_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        img_arr = self.data[idx]
        target = self.targets[idx]

        # NumPy dizisini PIL görseline çevir
        img = Image.fromarray(img_arr)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_cifar10_1_dataloader(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    download: bool = True,
) -> DataLoader:
    """
    CIFAR-10.1 için DataLoader döndürür.
    """
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    dataset = CIFAR10Point1Dataset(
        root=data_dir,
        transform=test_transform,
        download=download,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader

