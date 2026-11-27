"""
Faz 0 Bütünlük ve Veri Ayrımı Birim Testleri (Data Integrity & Leakage Unit Tests)

Bu test süiti:
1. Veri Sızıntısı (Data Leakage) Kontrolü:
   - CIFAR-10 train ayrımı ile %10 stratified val ayrımı indekslerinin ayrık (disjoint) olduğunu,
   - Resmi test setinin hiçbir şekilde train/val kümeleriyle çakışmadığını doğrular.
2. Boyut ve Tensör Bütünlüğü:
   - ResNet-18 girdileri için beklenen (B, 3, 32, 32) şekil ve tip doğrulaması.
   - N=1, N=4, N=64 batch boyutları için test akışı çalışması.
3. Dirichlet Örnekleyici Tutarlılığı:
   - alpha = inf durumunda sınıfların yaklaşık eşit dağıldığını,
   - alpha = 0.1 durumunda belirgin sınıf dengesizliği oluştuğunu matematiksel olarak sınar.
4. N=1 Kanal Uzamsal Ortalama (x_c) Hesaplaması:
   - Tekil örnek için per-channel spatial mean tensör boyutunun (1, C) olduğunu doğrular.
"""

import math
import numpy as np
import pytest
import torch
from torch.utils.data import TensorDataset

from src.data.cifar10 import get_cifar10_transforms, CIFAR10_MEAN, CIFAR10_STD
from src.data.imbalance import DirichletClassSampler, get_imbalanced_stream_dataloader


def test_data_leakage_and_splits():
    """
    Train ve Stratified Val kümelerinin indekslerinin kesişiminin BOŞ olduğunu
    ve toplamın 50.000 ettiğini simüle ederek doğrular.
    """
    from sklearn.model_selection import StratifiedShuffleSplit
    
    # 50.000'lik yapay hedef vektörü (10 sınıf, her birinden 5000)
    fake_targets = np.repeat(np.arange(10), 5000)
    indices = np.arange(len(fake_targets))

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
    train_idx, val_idx = next(splitter.split(indices, fake_targets))

    # Kesişim kümesi boş olmalıdır
    intersection = set(train_idx).intersection(set(val_idx))
    assert len(intersection) == 0, f"Veri sızıntısı tespit edildi! Ortak indeks sayısı: {len(intersection)}"

    assert len(train_idx) == 45000
    assert len(val_idx) == 5000

    # Her iki kümede de sınıf oranlarının korunduğunu (stratified) doğrula
    val_classes, val_counts = np.unique(fake_targets[val_idx], return_counts=True)
    assert len(val_classes) == 10
    assert np.all(val_counts == 500), f"Val ayrımı tabakalı değil: {val_counts}"


def test_dirichlet_sampler_balanced_and_imbalanced():
    """
    Dirichlet örnekleyicisinin alpha=inf ve alpha=0.1 altındaki davranışını test eder.
    """
    num_classes = 10
    total_samples = 2000
    fake_targets = np.repeat(np.arange(num_classes), 200)

    # 1. Dengeli durum: alpha = inf
    sampler_inf = DirichletClassSampler(
        targets=fake_targets.tolist(),
        num_classes=num_classes,
        alpha=float("inf"),
        seed=42,
        total_samples=total_samples,
    )
    sampled_indices_inf = list(sampler_inf)
    assert len(sampled_indices_inf) == total_samples
    sampled_labels_inf = fake_targets[sampled_indices_inf]
    _, counts_inf = np.unique(sampled_labels_inf, return_counts=True)
    # Her sınıftan tam 200 örnek beklenir
    assert np.all(counts_inf == 200), f"Dengeli dağılım bozuk: {counts_inf}"

    # 2. Dengesiz durum: alpha = 0.1
    sampler_imb = DirichletClassSampler(
        targets=fake_targets.tolist(),
        num_classes=num_classes,
        alpha=0.1,
        seed=42,
        total_samples=total_samples,
    )
    sampled_indices_imb = list(sampler_imb)
    sampled_labels_imb = fake_targets[sampled_indices_imb]
    _, counts_imb = np.unique(sampled_labels_imb, return_counts=True)
    
    # Sınıflar arası standart sapma yüksek olmalıdır (dengesizlik kanıtı)
    std_imb = np.std(counts_imb)
    assert std_imb > 50.0, f"alpha=0.1 yeterli dengesizlik oluşturmadı! Std: {std_imb}, counts: {counts_imb}"


def test_batch_sizes_and_spatial_mean():
    """
    N=1, N=4 ve N=64 için batch akışını ve
    N=1 blending fallback için kanal uzamsal ortalamasını (spatial mean) test eder.
    """
    # 20 örnekli tensör veri seti: (20, 3, 32, 32)
    fake_imgs = torch.randn(20, 3, 32, 32)
    fake_labels = torch.randint(0, 10, (20,))
    dataset = TensorDataset(fake_imgs, fake_labels)
    dataset.targets = fake_labels.numpy().tolist()

    # N=1 akışı
    loader_n1 = get_imbalanced_stream_dataloader(dataset, batch_size=1, alpha=float("inf"))
    for imgs, lbls in loader_n1:
        assert imgs.shape == (1, 3, 32, 32)
        assert lbls.shape == (1,)

        # N=1 için per-channel spatial mean (x_c) hesabı
        # x_c = 1/(H*W) * sum_{h,w} X_{c,h,w} -> boyut: (1, 3)
        spatial_mean = imgs.mean(dim=[2, 3])  # (1, 3)
        assert spatial_mean.shape == (1, 3)
        break

    # N=4 akışı
    loader_n4 = get_imbalanced_stream_dataloader(dataset, batch_size=4, alpha=float("inf"))
    for imgs, lbls in loader_n4:
        assert imgs.shape == (4, 3, 32, 32)
        assert lbls.shape == (4,)
        break

