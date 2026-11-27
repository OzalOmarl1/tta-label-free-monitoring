"""
ResNet-18 CIFAR-10 Modeli

CIFAR-10 (32x32 görseller) için uyarlanmış standart ResNet-18 mimarisi:
- Giriş katmanında ImageNet için olan 7x7 conv yerine 3x3 conv (stride=1, padding=1) kullanılır.
- İlk maxpool katmanı kaldırılmıştır (32x32 boyutunun erkenden 4x4'e küçülmesini engellemek için).
- Batch Normalization katmanları TTA (TENT, BN-adapt) uyumludur.
"""

import torch
import torch.nn as nn
import torchvision.models as models


def get_cifar_resnet18(num_classes: int = 10, pretrained: bool = False) -> nn.Module:
    """
    CIFAR-10 için optimize edilmiş ResNet-18 modelini döndürür.
    """
    model = models.resnet18(weights=None)
    
    # 32x32 CIFAR uyarlaması
    model.conv1 = nn.Conv2d(
        in_channels=3,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    # İlk max pooling katmanını etkisiz hale getir (identity)
    model.maxpool = nn.Identity()
    
    # Son tam bağlantılı (fc) katmanı 10 sınıfa eşle
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

