"""
Etiketsiz doğruluk tahmin edicileri (label-free accuracy estimators).

Hedef alan için YALNIZCA etiketsiz logitler kullanılır. Referans (kalibrasyon) bilgisi,
kaynak dağılımından ayrılmış etiketli doğrulama kümesinden (5k, eğitimde görülmemiş) gelir.

  AC      : Ortalama güven (Hendrycks & Gimpel, 2017 taban çizgisi).
  AC-TS   : Doğrulama kümesinde NLL ile öğrenilen sıcaklık ölçekleme sonrası ortalama güven
            (Guo ve ark., 2017).
  DoC     : Güven farkı — acc_val - (AC_val - AC_hedef)  (Guillory ve ark., 2021).
  ATC-MC  : Ortalama eşiklenmiş güven, skor = maksimum olasılık (Garg ve ark., 2022).
  ATC-NE  : Aynı, skor = negatif entropi (Garg ve ark., 2022).
  GDE     : Bağımsız eğitilmiş diğer modellerle tahmin uyuşması
            (Jiang ve ark., 2022; burada aynı TTA yöntemiyle adapte edilmiş modeller arası).

Tümü yüzde (%) cinsinden doğruluk tahmini döndürür.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import minimize_scalar

from .metrics import softmax_np


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    lg = logits.astype(np.float64)

    def nll_T(logT):
        z = lg / np.exp(logT)
        z = z - z.max(1, keepdims=True)
        lse = np.log(np.exp(z).sum(1))
        return float((lse - z[np.arange(len(y)), y]).mean())

    res = minimize_scalar(nll_T, bounds=(np.log(0.05), np.log(20.0)), method="bounded")
    return float(np.exp(res.x))


def _neg_entropy(p: np.ndarray) -> np.ndarray:
    return (p * np.log(np.clip(p, 1e-12, 1.0))).sum(1)


class Reference:
    """Doğrulama kümesinden türetilen kalibrasyon bilgisi (eşikler, sıcaklık, güven)."""

    def __init__(self, val_logits: np.ndarray, val_y: np.ndarray):
        p = softmax_np(val_logits)
        self.acc = float((p.argmax(1) == val_y).mean())
        self.ac = float(p.max(1).mean())
        self.T = fit_temperature(val_logits, val_y)
        # ATC eşiği: P(skor > t) = val doğruluğu olacak şekilde
        q = 1.0 - self.acc
        self.t_mc = float(np.quantile(p.max(1), q))
        self.t_ne = float(np.quantile(_neg_entropy(p), q))


def estimate_all(logits: np.ndarray, ref: Reference,
                 other_preds: Optional[List[np.ndarray]] = None) -> Dict[str, float]:
    p = softmax_np(logits)
    conf = p.max(1)
    out = {
        "AC": 100.0 * conf.mean(),
        "AC-TS": 100.0 * softmax_np(logits, ref.T).max(1).mean(),
        "DoC": 100.0 * (ref.acc - (ref.ac - conf.mean())),
        "ATC-MC": 100.0 * (conf > ref.t_mc).mean(),
        "ATC-NE": 100.0 * (_neg_entropy(p) > ref.t_ne).mean(),
    }
    if other_preds:
        pred = p.argmax(1)
        out["GDE"] = 100.0 * float(np.mean([(pred == o).mean() for o in other_preds]))
    return {k: float(v) for k, v in out.items()}


CONF_ESTIMATORS = ["AC", "AC-TS", "DoC", "ATC-MC", "ATC-NE"]
ALL_ESTIMATORS = CONF_ESTIMATORS + ["GDE"]
