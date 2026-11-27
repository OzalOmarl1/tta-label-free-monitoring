"""
CIFAR-10 Veri Kümesi Modülü (HuggingFace Parquet ve Resmi Kaynak Desteği)

Bu modül:
1. İndirilmiş parquet veya standart CIFAR-10 verisini yükler.
2. 50.000 eğitim örneğinden %10 tabakalı (stratified) kaynak doğrulama (source-val: 5.000) ayırır.
3. Test kümesini (10.000 örnek) kesinlikle yalıtır.
4. Standart normalizasyonu uygular.
"""

import os
import io
from typing import Tuple, Optional, Callable
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import StratifiedShuffleSplit

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)

CIFAR10_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
)


class ParquetCIFAR10Dataset(Dataset):
    """
    Parquet formatında saklanan CIFAR-10 veri kümesini okuyan Dataset sınıfı.
    """

    def __init__(self, parquet_path: str, transform: Optional[Callable] = None):
        if not os.path.exists(parquet_path):
            raise FileNotFoundError(f"Parquet dosyası bulunamadı: {parquet_path}")
        
        self.df = pd.read_parquet(parquet_path)
        self.transform = transform
        self.targets = self.df["label"].to_numpy(dtype=np.int64)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_dict = row["img"]
        img_bytes = img_dict["bytes"]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        target = self.targets[idx]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def get_cifar10_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_transform, test_transform


def get_cifar10_datasets(
    data_dir: str = "./data",
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[Dataset, Dataset, Dataset]:
    """
    CIFAR-10 train, source-val ve test kümelerini döndürür.
    Öncelikle parquet dosyalarını arar, yoksa torchvision CIFAR10 arayüzüne döner.
    """
    train_transform, test_transform = get_cifar10_transforms()

    train_parquet = os.path.join(data_dir, "cifar10_train.parquet")
    test_parquet = os.path.join(data_dir, "cifar10_test.parquet")

    if os.path.exists(train_parquet) and os.path.exists(test_parquet):
        # Parquet üzerinden yükle
        raw_train_dataset = ParquetCIFAR10Dataset(train_parquet, transform=train_transform)
        raw_val_dataset = ParquetCIFAR10Dataset(train_parquet, transform=test_transform)
        test_dataset = ParquetCIFAR10Dataset(test_parquet, transform=test_transform)
        targets = raw_train_dataset.targets
    else:
        from torchvision import datasets
        raw_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
        raw_val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=test_transform)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transform)
        targets = np.array(raw_train_dataset.targets)

    indices = np.arange(len(targets))
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(indices, targets))

    train_dataset = Subset(raw_train_dataset, train_idx)
    val_dataset = Subset(raw_val_dataset, val_idx)

    # Hedef özniteliklerini Subset üzerine ekle (DirichletSampler için)
    train_dataset.targets = [int(targets[i]) for i in train_idx]
    val_dataset.targets = [int(targets[i]) for i in val_idx]

    return train_dataset, val_dataset, test_dataset


def get_cifar10_dataloaders(
    data_dir: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_set, val_set, test_set = get_cifar10_datasets(data_dir=data_dir, val_ratio=val_ratio, seed=seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader
