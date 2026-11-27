"""Etiketli değerlendirme metrikleri (yalnızca DEĞERLENDİRMEDE kullanılır, adaptasyonda asla)."""

from __future__ import annotations

import numpy as np


def softmax_np(logits: np.ndarray, T: float = 1.0) -> np.ndarray:
    z = logits.astype(np.float64) / T
    z -= z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def accuracy(probs: np.ndarray, y: np.ndarray) -> float:
    return float((probs.argmax(1) == y).mean() * 100.0)


def ece(probs: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """Eşit genişlikli 15 kutulu Beklenen Kalibrasyon Hatası (Guo ve ark., 2017), yüzde."""
    conf = probs.max(1)
    correct = (probs.argmax(1) == y).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += abs(conf[m].mean() - correct[m].mean()) * m.mean()
    return float(total * 100.0)


def nll(probs: np.ndarray, y: np.ndarray) -> float:
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1.0)).mean())


def brier(probs: np.ndarray, y: np.ndarray) -> float:
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y)), y] = 1.0
    return float(((probs - onehot) ** 2).sum(1).mean())


def mean_conf(probs: np.ndarray) -> float:
    return float(probs.max(1).mean() * 100.0)


def mean_entropy(probs: np.ndarray) -> float:
    return float(-(probs * np.log(np.clip(probs, 1e-12, 1.0))).sum(1).mean())


def all_metrics(logits: np.ndarray, y: np.ndarray) -> dict:
    p = softmax_np(logits)
    acc = accuracy(p, y)
    mc = mean_conf(p)
    return {"acc": acc, "ece": ece(p, y), "nll": nll(p, y), "brier": brier(p, y),
            "mean_conf": mc, "overconf": mc - acc, "mean_entropy": mean_entropy(p),
            "n": int(len(y))}
