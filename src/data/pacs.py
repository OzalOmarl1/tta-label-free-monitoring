"""
PACS Stil/Alan Kayması (Style/Domain Shift) Veri Kümesi Modülü

PACS Veri Kümesi (Li et al., ICCV 2017):
4 farklı stil/alan (domain) içerir:
- Photo (P)
- Art Painting (A)
- Cartoon (C)
- Sketch (S)

7 kategori (sınıf):
dog, elephant, giraffe, guitar, horse, house, person

Bu modül:
1. PACS veri kümesinin dizin yapısını veya standart ayrım dosyalarını (splits) okur.
2. 32x32 boyutuna yeniden boyutlandırma (ResNet-18 CIFAR mimarisiyle tam uyum) ve
   normalizasyon seçenekleri sunar.
3. PyTorch Dataset ve DataLoader arayüzü sunar.
"""

import os
from typing import Optional, Callable, List
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from src.data.cifar10 import CIFAR10_MEAN, CIFAR10_STD

PACS_DOMAINS = ["photo", "art_painting", "cartoon", "sketch"]
PACS_CLASSES = ["dog", "elephant", "giraffe", "guitar", "horse", "house", "person"]


class PACSDataset(Dataset):
    """
    PACS Stil/Alan Kayması Dataset sınıfı.
    Görseller dizin formatında (root/domain/class/image.jpg) veya liste formatında okunabilir.
    """

    def __init__(
        self,
        domain: str,
        root: str = "./data/PACS",
        transform: Optional[Callable] = None,
        image_size: int = 32,
    ):
        """
        Args:
            domain: 'photo', 'art_painting', 'cartoon', veya 'sketch'
            root: PACS verisinin bulunduğu kök dizin
            transform: İsteğe bağlı görsel dönüşümleri
            image_size: CIFAR-10 uyumlu uzamsal boyut (varsayılan: 32)
        """
        domain_lower = domain.lower()
        if domain_lower not in PACS_DOMAINS:
            raise ValueError(f"Geçersiz PACS domain: '{domain}'. Geçerli olanlar: {PACS_DOMAINS}")

        self.domain = domain_lower
        self.root = root
        self.image_size = image_size
        self.transform = transform

        self.samples: List[tuple] = []
        self._load_samples()

    def _load_samples(self) -> None:
        """Dizin hiyerarşisini tarayarak görsel yollarını ve etiketleri yükler."""
        domain_dir = os.path.join(self.root, self.domain)
        if not os.path.exists(domain_dir):
            # Dosya sistemi büyük/küçük harf duyarlılığı için kontrol
            alt_dirs = [d for d in os.listdir(self.root) if d.lower() == self.domain] if os.path.exists(self.root) else []
            if alt_dirs:
                domain_dir = os.path.join(self.root, alt_dirs[0])
            else:
                return

        for class_idx, class_name in enumerate(PACS_CLASSES):
            class_dir = os.path.join(domain_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fpath = os.path.join(class_dir, fname)
                    self.samples.append((fpath, class_idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fpath, target = self.samples[idx]
        with open(fpath, "rb") as f:
            img = Image.open(f).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_pacs_dataloader(
    domain: str,
    root: str = "./data/PACS",
    batch_size: int = 128,
    num_workers: int = 2,
    image_size: int = 32,
) -> DataLoader:
    """
    PACS belirli bir stil/alan için DataLoader döndürür.
    Görselleri CIFAR-10 uyumlu 32x32 boyutuna çeker ve standart CIFAR-10 normalizasyonu uygular.
    """
    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    dataset = PACSDataset(
        domain=domain,
        root=root,
        transform=test_transform,
        image_size=image_size,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader

