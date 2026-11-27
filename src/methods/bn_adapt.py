"""
Test-Time Batch Normalization Adaptation (BN-adapt) Yöntem Modülü

Schneider et al. (NeurIPS 2020):
- Model ağırlıklarını ve afin parametrelerini tamamen dondurur.
- Hiçbir etiket veya optimizasyon/gradyan adımı kullanmaz (label-free, gradient-free).
- Test partisi üzerinden anlık ortalama ve varyansı hesaplayarak BatchNorm katmanlarını günceller.
- N=1 durumunda teklifteki kayıtlı kural:
  * Varyans kaynakta dondurulur: sigma_blend^2 = sigma_source^2
  * x: tekil örneğin kanal uzamsal ortalaması (per-channel spatial mean)
  * Ortalama harmanlanır: mu_blend = lambda * mu_source + (1 - lambda) * x
  * lambda sabittir (önceden belirlenmiş pre-registered değer, örn. 0.9 [VERIFY]).
"""

from typing import Optional
import torch
import torch.nn as nn


class BlendedBatchNorm2d(nn.Module):
    """
    N=1 durumunda per-channel spatial mean ve dondurulmuş kaynak varyansı ile
    harmanlama (blending fallback) yapan sarmalayıcı BatchNorm katmanı.
    N > 1 durumunda standart BatchNorm2d anlık parti istatistiklerini kullanır.
    """

    def __init__(self, original_bn: nn.BatchNorm2d, blend_lambda: float = 0.9):
        super().__init__()
        self.num_features = original_bn.num_features
        self.eps = original_bn.eps
        self.blend_lambda = blend_lambda

        # Kaynak istatistikleri ve afin parametreler
        self.register_buffer("running_mean", original_bn.running_mean.clone().detach())
        self.register_buffer("running_var", original_bn.running_var.clone().detach())
        
        if original_bn.weight is not None:
            self.weight = nn.Parameter(original_bn.weight.clone())
        else:
            self.register_parameter("weight", None)

        if original_bn.bias is not None:
            self.bias = nn.Parameter(original_bn.bias.clone())
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape

        if b == 1:
            # N=1 Blending Fallback:
            # 1. x_c: Tekil örneğin kanal uzamsal ortalaması (1, C, 1, 1)
            spatial_mean = x.mean(dim=[2, 3], keepdim=True)
            
            # 2. mu_blend: lambda * mu_source + (1 - lambda) * x_c
            source_mean = self.running_mean.view(1, c, 1, 1)
            mu_blend = self.blend_lambda * source_mean + (1.0 - self.blend_lambda) * spatial_mean

            # 3. sigma_blend^2: Kaynak varyansı dondurulur
            sigma_blend_sq = self.running_var.view(1, c, 1, 1)

            # Normalizasyon
            x_norm = (x - mu_blend) / torch.sqrt(sigma_blend_sq + self.eps)

            # Afin ölçekleme
            if self.weight is not None:
                x_norm = x_norm * self.weight.view(1, c, 1, 1)
            if self.bias is not None:
                x_norm = x_norm + self.bias.view(1, c, 1, 1)
            return x_norm
        else:
            # N > 1 için standart Test-Time BN: Anlık partinin ortalama ve varyansı
            batch_mean = x.mean(dim=[0, 2, 3], keepdim=True)
            batch_var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)

            x_norm = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            if self.weight is not None:
                x_norm = x_norm * self.weight.view(1, c, 1, 1)
            if self.bias is not None:
                x_norm = x_norm + self.bias.view(1, c, 1, 1)
            return x_norm


def convert_to_blended_bn(model: nn.Module, blend_lambda: float = 0.9) -> nn.Module:
    """
    Model içerisindeki tüm BatchNorm2d katmanlarını N=1 harmanlamasını destekleyen
    BlendedBatchNorm2d katmanlarına dönüştürür.
    """
    for name, module in model.named_children():
        if isinstance(module, nn.BatchNorm2d):
            setattr(model, name, BlendedBatchNorm2d(module, blend_lambda=blend_lambda))
        else:
            convert_to_blended_bn(module, blend_lambda=blend_lambda)
    return model


class BNAdaptAdapter:
    """
    BN-adapt adaptasyon yöneticisi.
    Kesinlikle etiket (target) almaz, gradyan hesaplamaz.
    """

    def __init__(self, model: nn.Module, blend_lambda: float = 0.9):
        self.model = convert_to_blended_bn(model, blend_lambda=blend_lambda)
        self.model.eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gelen test partisini normalize ederek tahmin üretir.
        """
        return self.model(x)

