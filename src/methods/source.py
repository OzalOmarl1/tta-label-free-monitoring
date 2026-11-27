"""
Kaynak Model (Source / No-Adaptation) Yöntem Modülü

Bu yöntem, kaynak alanda eğitilmiş orijinal modeli hiçbir adaptasyon,
istatistik güncellemesi veya parametre modifikasyonu yapmadan doğrudan
test verisine uygular (Temel Çizgi / Baseline).
"""

import torch
import torch.nn as nn


class SourceAdapter:
    """
    Kaynak Model (Adaptasyonsuz) Çıkarım Yöneticisi
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Dondurulmuş kaynak modeliyle doğrudan ileri yayılım yapar.
        """
        return self.model(x)

