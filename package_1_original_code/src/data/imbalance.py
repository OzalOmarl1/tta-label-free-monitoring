"""
Dirichlet Sınıf Dengesizliği (Class Imbalance) ve Akış Örnekleyicisi Modülü

Bu modül:
1. Simetrik Dirichlet dağılımından sınıf olasılık vektörü p ~ Dir(alpha * 1_C) üretir.
   - alpha = infinity: Tamamen dengeli sınıf dağılımı (uniform p_c = 1/C)
   - alpha = 0.1: Aşırı sınıf dengesizliği (bazı sınıflar çok baskın, bazıları neredeyse sıfır)
2. Verilen bir test veri kümesini (CIFAR-10, CIFAR-10.1, CIFAR-10-C vb.) bu olasılık vektörüne
   göre örnekleyerek veya yeniden ağırlıklandırarak test akışı (test stream) oluşturur.
3. Değişken parti boyutlarını (batch size N in {1, 4, 16, 64, 200}) destekler.
"""

import math
from typing import List, Optional, Tuple, Iterator
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler, DataLoader


class DirichletClassSampler(Sampler[int]):
    """
    Belirli bir Dirichlet konsantrasyon katsayısı (alpha) ile sınıfları örnekleyen
    PyTorch Sampler sınıfı.
    
    Protokol:
    - alpha == float('inf') veya alpha >= 1e4: Dengeli dağılım (her sınıftan eşit sayıda)
    - alpha < 1e4: p ~ Dirichlet(alpha * ones(num_classes)) çekilir.
      Toplam len(dataset) kadar örnek bu dağılıma göre seçilir.
    """

    def __init__(
        self,
        targets: List[int],
        num_classes: int = 10,
        alpha: float = float("inf"),
        seed: int = 42,
        total_samples: Optional[int] = None,
    ):
        """
        Args:
            targets: Veri kümesindeki tüm örneklerin etiket listesi.
            num_classes: Toplam sınıf sayısı (CIFAR için 10).
            alpha: Dirichlet konsantrasyon katsayısı (0.1, 1.0, 10.0, inf vb.).
            seed: Tekrarlanabilirlik tohumu.
            total_samples: Üretilecek toplam örnek sayısı (None ise len(targets)).
        """
        super().__init__()
        self.targets = np.array(targets)
        self.num_classes = num_classes
        self.alpha = alpha
        self.seed = seed
        self.total_samples = total_samples if total_samples is not None else len(self.targets)

        # Sınıflara ait indeksleri grupla
        self.class_indices = {
            c: np.where(self.targets == c)[0]
            for c in range(self.num_classes)
        }

        self.indices = self._generate_indices()

    def _generate_indices(self) -> List[int]:
        rng = np.random.default_rng(self.seed)

        if math.isinf(self.alpha) or self.alpha >= 1e4:
            # Dengeli dağılım: sınıflar kesinlikle eşit sayıda temsil edilir
            base_count = self.total_samples // self.num_classes
            rem = self.total_samples % self.num_classes
            class_counts = np.full(self.num_classes, base_count, dtype=int)
            if rem > 0:
                class_counts[:rem] += 1
        else:
            # Dirichlet dağılımından sınıf olasılıkları çek
            class_probs = rng.dirichlet(np.ones(self.num_classes) * self.alpha)
            # Belirlenen olasılıklara göre toplam örnek sayısını sınıflara paylaştır
            class_counts = rng.multinomial(self.total_samples, class_probs)

        sampled_indices = []
        for c in range(self.num_classes):
            n_c = class_counts[c]
            if n_c > 0:
                available = self.class_indices[c]
                if len(available) == 0:
                    continue
                # İhtiyaç varsa yerine koyarak (with replacement) örnekle
                replace = n_c > len(available)
                chosen = rng.choice(available, size=n_c, replace=replace)
                sampled_indices.extend(chosen.tolist())

        # Örnekleme akışını karıştır (stream ordering)
        rng.shuffle(sampled_indices)
        return sampled_indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


def get_imbalanced_stream_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    alpha: float = float("inf"),
    num_classes: int = 10,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """
    Belirli bir parti boyutu (N) ve Dirichlet dengesizliği (alpha) ile yapılandırılmış
    test akışı DataLoader'ı döndürür.
    
    Özellikle N=1 için batch_size=1, drop_last=False olarak ayarlanır.
    """
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise AttributeError("Dataset 'targets' listesi/dizisi içermelidir.")

    sampler = DirichletClassSampler(
        targets=targets,
        num_classes=num_classes,
        alpha=alpha,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
