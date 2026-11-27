"""
TTA yöntemleri: Source, BN-adapt, TENT.

Orijinal kod tabanındaki BlendedBatchNorm2d (N>1: anlık parti istatistikleri; N=1: kaynak
istatistiklerine harmanlama) aynen korunmuştur.

DÜZELTME (AUDIT A5): Orijinal TentAdapter, güncelleme adımından SONRA aynı parti üzerinde
ikinci bir ileri geçiş yapıp tahmini ondan döndürüyordu. Resmi Tent uygulamasında
(Wang ve ark., 2021) tahmin, kaybın hesaplandığı ileri geçişin çıktısıdır (güncelleme
yalnızca SONRAKİ partileri etkiler). Burada `predict="official"` varsayılandır; orijinal
davranış `predict="after_update"` ile ablasyon olarak ölçülür.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class BlendedBatchNorm2d(nn.Module):
    def __init__(self, bn: nn.BatchNorm2d, blend_lambda: float = 0.9):
        super().__init__()
        self.num_features = bn.num_features
        self.eps = bn.eps
        self.blend_lambda = blend_lambda
        self.register_buffer("running_mean", bn.running_mean.clone().detach())
        self.register_buffer("running_var", bn.running_var.clone().detach())
        self.weight = nn.Parameter(bn.weight.detach().clone())
        self.bias = nn.Parameter(bn.bias.detach().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c = x.shape[0], x.shape[1]
        if b == 1:
            mu = self.blend_lambda * self.running_mean.view(1, c, 1, 1) + \
                 (1.0 - self.blend_lambda) * x.mean(dim=[2, 3], keepdim=True)
            var = self.running_var.view(1, c, 1, 1)
            x = (x - mu) / torch.sqrt(var + self.eps)
            return x * self.weight.view(1, c, 1, 1) + self.bias.view(1, c, 1, 1)
        # N>1: anlık parti ortalaması ve yanlı (biased) varyans ile normalizasyon. Orijinal elle yazılmış
        # hesapla matematiksel olarak özdeştir; hız için birleşik cuDNN çekirdeği kullanılır
        # (eşdeğerlik tests/test_core.py::test_fused_bn_equals_manual ile sınanır).
        return F.batch_norm(x, None, None, self.weight, self.bias, training=True, momentum=0.0, eps=self.eps)


def convert_to_blended_bn(model: nn.Module, blend_lambda: float = 0.9) -> nn.Module:
    for name, m in model.named_children():
        if isinstance(m, nn.BatchNorm2d):
            setattr(model, name, BlendedBatchNorm2d(m, blend_lambda))
        else:
            convert_to_blended_bn(m, blend_lambda)
    return model


def collect_bn_affine(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    params, names = [], []
    for nm, m in model.named_modules():
        if isinstance(m, BlendedBatchNorm2d):
            params += [m.weight, m.bias]
            names += [f"{nm}.weight", f"{nm}.bias"]
    return params, names


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(1) * logits.log_softmax(1)).sum(1)


class SourceAdapter:
    name = "Source"

    def __init__(self, model: nn.Module):
        self.model = model.eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class BNAdaptAdapter:
    name = "BN-adapt"

    def __init__(self, model: nn.Module, blend_lambda: float = 0.9):
        self.model = convert_to_blended_bn(model, blend_lambda).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class TentAdapter:
    name = "TENT"

    def __init__(self, model: nn.Module, lr: float = 1e-3, steps: int = 1,
                 predict: str = "official", blend_lambda: float = 0.9):
        assert predict in ("official", "after_update")
        self.model = convert_to_blended_bn(model, blend_lambda).eval().requires_grad_(False)
        params, _ = collect_bn_affine(self.model)
        for p in params:
            p.requires_grad_(True)
        self.optimizer = optim.Adam(params, lr=lr, betas=(0.9, 0.999), weight_decay=0.0)
        self.steps = steps
        self.predict = predict

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        out = None
        for _ in range(self.steps):
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x)
            loss = softmax_entropy(out).mean()
            loss.backward()
            self.optimizer.step()
        if self.predict == "after_update":
            with torch.no_grad():
                out = self.model(x)
        return out.detach()


def build_adapter(method: str, model: nn.Module, **kw):
    if method == "Source":
        return SourceAdapter(model)
    if method == "BN-adapt":
        return BNAdaptAdapter(model)
    if method == "TENT":
        return TentAdapter(model, lr=kw.get("lr", 1e-3), steps=kw.get("steps", 1),
                           predict=kw.get("predict", "official"))
    raise ValueError(method)
