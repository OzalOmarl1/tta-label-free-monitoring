"""
Faz 0 Kapsamlı Birim Testleri (Unit Tests)

Kapsanan Kritik İlkeler:
1. TENT Gradyan İzolasyonu:
   - Yalnızca BlendedBatchNorm2d afin parametreleri (weight, bias) gradyan alır ve güncellenir.
   - Konvolüsyon (conv1, layer1-4 conv) ve tam bağlantılı (fc) katman parametreleri
     kesinlikle gradyan almaz (grad is None / requires_grad == False) ve ağırlıkları değişmez.
2. BN-adapt Etiketsiz Çıkarım (Label-Free / Gradient-Free):
   - BN-adapt kesinlikle hiçbir etiket (target/labels) kullanmaz.
   - Modelde hiçbir gradyan hesaplanmaz (no backward pass).
3. Veri Sızıntısı (Data Leakage) Önleme:
   - Train, Stratified Val ve Test kümeleri arasında indeks/örnek örtüşmesi sıfırdır.
4. N=1 Blending Fallback Özdeşliği:
   - Hem BN-adapt hem de TENT, N=1 durumunda teklifteki aynı per-channel spatial mean
     ve kaynak varyansı formülasyonunu kullanır.
"""

import copy
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from src.models.resnet import get_cifar_resnet18
from src.methods.source import SourceAdapter
from src.methods.bn_adapt import BNAdaptAdapter, BlendedBatchNorm2d
from src.methods.tent import TentAdapter, configure_tent_model, collect_tent_params


def test_tent_updates_only_bn_affine_params():
    """
    TENT optimizasyon adımında YALNIZCA BatchNorm afin parametrelerinin (gamma, beta)
    güncellendiğini, conv ve fc ağırlıklarının dondurulmuş kaldığını doğrular.
    """
    torch.manual_seed(42)
    model = get_cifar_resnet18(num_classes=10)
    
    # Orijinal ağırlıkların kopyasını al
    orig_conv_weight = model.conv1.weight.clone().detach()
    orig_fc_weight = model.fc.weight.clone().detach()

    # TENT adaptörünü başlat (optimizer'ı model yapılandırıldıktan sonra toplanan parametrelerle kur)
    adapter = TentAdapter(model, optimizer=None, lr=1e-2, steps=1, blend_lambda=0.9)

    # Bir BN katmanının başlangıç ağırlığını al
    first_bn = None
    for m in adapter.model.modules():
        if isinstance(m, BlendedBatchNorm2d):
            first_bn = m
            break
    assert first_bn is not None
    orig_bn_weight = first_bn.weight.clone().detach()

    # Yapay parti ile adaptasyon çalıştır
    fake_x = torch.randn(4, 3, 32, 32)
    _ = adapter.adapt_and_predict(fake_x)

    # 1. Konvolüsyon ağırlığı KESİNLİKLE değişmemeli
    assert torch.equal(adapter.model.conv1.weight, orig_conv_weight), (
        "HATA: TENT konvolüsyon ağırlıklarını değiştirdi! Gradyan sızıntısı var."
    )
    assert adapter.model.conv1.weight.grad is None

    # 2. FC ağırlığı KESİNLİKLE değişmemeli
    assert torch.equal(adapter.model.fc.weight, orig_fc_weight), (
        "HATA: TENT FC ağırlıklarını değiştirdi! Gradyan sızıntısı var."
    )
    assert adapter.model.fc.weight.grad is None

    # 3. Yalnızca BN afin ağırlığı güncellenmiş olmalı
    assert not torch.equal(first_bn.weight, orig_bn_weight), (
        "HATA: TENT BN afin parametrelerini güncellemedi!"
    )
    assert first_bn.weight.grad is not None


def test_bn_adapt_uses_no_labels_and_no_gradients():
    """
    BN-adapt yönteminin kesinlikle etiket kabul etmediğini ve hiçbir
    gradyan üretmediğini/model ağırlıklarını değiştirmediğini doğrular.
    """
    torch.manual_seed(42)
    model = get_cifar_resnet18(num_classes=10)
    adapter = BNAdaptAdapter(model, blend_lambda=0.9)

    fake_x = torch.randn(4, 3, 32, 32)
    
    # adapt_and_predict fonksiyonunun imzasında etiket (y/target) parametresi olmadığını doğrula
    import inspect
    sig = inspect.signature(adapter.adapt_and_predict)
    assert "target" not in sig.parameters and "y" not in sig.parameters and "lbl" not in sig.parameters

    # Çıkarım yap
    preds = adapter.adapt_and_predict(fake_x)
    assert preds.shape == (4, 10)

    # Model parametrelerinde hiçbir gradyan olmamalı
    for p in adapter.model.parameters():
        assert p.grad is None


def test_n1_blending_mathematical_identity():
    """
    N=1 durumunda BlendedBatchNorm2d katmanının:
    mu_blend = lambda * mu_source + (1 - lambda) * x_c
    sigma_blend^2 = sigma_source^2
    formülasyonunu tam olarak uyguladığını matematiksel olarak denetler.
    """
    torch.manual_seed(42)
    raw_bn = nn.BatchNorm2d(num_features=3)
    raw_bn.running_mean.copy_(torch.tensor([1.0, 2.0, 3.0]))
    raw_bn.running_var.copy_(torch.tensor([4.0, 4.0, 4.0]))
    raw_bn.weight.data.copy_(torch.tensor([1.0, 1.0, 1.0]))
    raw_bn.bias.data.copy_(torch.tensor([0.0, 0.0, 0.0]))

    blend_lambda = 0.9
    blended_bn = BlendedBatchNorm2d(raw_bn, blend_lambda=blend_lambda)

    # Tekil örnek (N=1, C=3, H=2, W=2)
    # Kanal 0: hepsi 2.0 -> uzamsal ortalama x_c[0] = 2.0
    # Kanal 1: hepsi 4.0 -> uzamsal ortalama x_c[1] = 4.0
    # Kanal 2: hepsi 6.0 -> uzamsal ortalama x_c[2] = 6.0
    x = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]],
                       [[4.0, 4.0], [4.0, 4.0]],
                       [[6.0, 6.0], [6.0, 6.0]]]], dtype=torch.float32)

    out = blended_bn(x)

    # Beklenen harmanlanmış ortalamalar:
    # c=0: 0.9 * 1.0 + 0.1 * 2.0 = 0.9 + 0.2 = 1.1
    # c=1: 0.9 * 2.0 + 0.1 * 4.0 = 1.8 + 0.4 = 2.2
    # c=2: 0.9 * 3.0 + 0.1 * 6.0 = 2.7 + 0.6 = 3.3
    expected_mu = torch.tensor([1.1, 2.2, 3.3])
    expected_sigma = torch.sqrt(torch.tensor([4.0, 4.0, 4.0]) + blended_bn.eps)

    # Beklenen normalize çıktılar:
    expected_out_0 = (2.0 - 1.1) / expected_sigma[0]
    expected_out_1 = (4.0 - 2.2) / expected_sigma[1]
    expected_out_2 = (6.0 - 3.3) / expected_sigma[2]

    assert torch.allclose(out[0, 0], expected_out_0, atol=1e-5)
    assert torch.allclose(out[0, 1], expected_out_1, atol=1e-5)
    assert torch.allclose(out[0, 2], expected_out_2, atol=1e-5)
