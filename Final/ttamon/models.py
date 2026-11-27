"""CIFAR uyarlamalı ResNet-18 (3x3 conv1, maxpool yok) — orijinal kod tabanından aynen korunmuştur."""
import torch
import torch.nn as nn
import torchvision.models as models


def get_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


_STATE_CACHE = {}


def load_source_model(path: str, device: str = "cuda") -> nn.Module:
    """Checkpoint'i yükler; durum sözlüğü bellekte önbelleklenir (her koşumda diskten okuma yok)."""
    if path not in _STATE_CACHE:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        _STATE_CACHE[path] = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model = get_cifar_resnet18(10)
    model.load_state_dict(_STATE_CACHE[path])   # load_state_dict kopyalar; önbellek değişmez
    return model.to(device).eval()
