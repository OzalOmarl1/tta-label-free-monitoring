"""
TENT (Test-Time Entropy Minimization) Yöntem Modülü

Wang et al. (ICLR 2021):
- Modelin evrişim ve lineer ağırlıklarını dondurur.
- Yalnızca BlendedBatchNorm2d katmanlarındaki afin parametreleri (gamma, beta) eğitilebilir bırakır.
- Shannon Entropisini (H(p) = -sum p_c log p_c) kayıp fonksiyonu olarak minimize eder.
- N=1 durumunda BN-adapt ile BİREBİR ÖZDEŞ BlendedBatchNorm2d fallback yapısını kullanır.
"""

from typing import Tuple, List
import torch
import torch.nn as nn
import torch.optim as optim

from src.methods.bn_adapt import BlendedBatchNorm2d, convert_to_blended_bn


def configure_tent_model(model: nn.Module, blend_lambda: float = 0.9) -> nn.Module:
    """
    Modeli TENT için yapılandırır:
    1. BatchNorm katmanlarını BlendedBatchNorm2d ile değiştirir (N=1 özdeş fallback garantisi).
    2. Modelin tüm parametrelerinin gradyanlarını kapatır.
    3. Yalnızca BlendedBatchNorm2d katmanlarının afin parametrelerini (weight/bias) açar.
    """
    model = convert_to_blended_bn(model, blend_lambda=blend_lambda)
    model.eval()
    model.requires_grad_(False)

    for m in model.modules():
        if isinstance(m, BlendedBatchNorm2d):
            if m.weight is not None:
                m.weight.requires_grad_(True)
            if m.bias is not None:
                m.bias.requires_grad_(True)

    return model


def collect_tent_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """
    TENT tarafından optimize edilecek parametreleri (yalnızca BN gamma ve beta) toplar.
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, BlendedBatchNorm2d):
            for np, p in m.named_parameters():
                if np in ["weight", "bias"] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Tahmin olasılıklarının Shannon Entropisini hesaplar:
    H(p) = - sum_{c} p_c * log(p_c)
    """
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy.mean()


class TentAdapter:
    """
    TENT Test Zamanı Adaptasyonu Yöneticisi
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer = None,
        lr: float = 1e-3,
        steps: int = 1,
        blend_lambda: float = 0.9,
        predict: str = "official",
    ):
        # DÜZELTME (AUDIT A5): Orijinal kod tahmini HER ZAMAN güncellemeden sonra ikinci bir ileri geçişle
        # üretiyordu. Resmi Tent'te tahmin, kaybın hesaplandığı ileri geçişin çıktısıdır ("official").
        # Eski davranış predict="after_update" ile seçilebilir (Faz-1 yeniden üretimi için).
        assert predict in ("official", "after_update")
        self.predict = predict
        self.model = configure_tent_model(model, blend_lambda=blend_lambda)
        self.steps = steps
        
        # Eğer dışarıdan optimizer verilmediyse, yalnızca TENT BN parametreleriyle oluştur
        if optimizer is None:
            params, _ = collect_tent_params(self.model)
            self.optimizer = optim.Adam(params, lr=lr)
        else:
            self.optimizer = optimizer

    def adapt_and_predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Gelen test partisi üzerinde 'steps' kadar entropi minimizasyon adımı atar
        ve ardından test tahminini döndürür.
        """
        outputs = None
        for _ in range(self.steps):
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = softmax_entropy(outputs)
            loss.backward()
            self.optimizer.step()

        if self.predict == "after_update":
            with torch.no_grad():
                return self.model(x)
        return outputs.detach()
