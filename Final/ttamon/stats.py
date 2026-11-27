"""İstatistik yardımcıları: ortalama ± std, t-tabanlı GA, Wilcoxon + Holm, hiyerarşik bootstrap."""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np
from scipy import stats as sps


def mean_std(x: Sequence[float]) -> Dict[str, float]:
    a = np.asarray(x, dtype=np.float64)
    sd = float(a.std(ddof=1)) if len(a) > 1 else 0.0
    return {"mean": float(a.mean()), "std": sd, "n": int(len(a))}


def t_ci(x: Sequence[float], level: float = 0.95) -> List[float]:
    a = np.asarray(x, dtype=np.float64)
    n = len(a)
    if n < 2:
        return [float(a.mean()), float(a.mean())]
    h = sps.t.ppf(0.5 + level / 2, n - 1) * a.std(ddof=1) / np.sqrt(n)
    return [float(a.mean() - h), float(a.mean() + h)]


def wilcoxon_paired(x: Sequence[float], y: Sequence[float]) -> Dict[str, float]:
    """İki yönlü Wilcoxon işaretli sıra testi (x - y) + eşleştirilmiş sıra-biserial etki büyüklüğü."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    d = x - y
    nz = d[d != 0]
    if len(nz) == 0:
        return {"n": int(len(d)), "median_diff": 0.0, "mean_diff": 0.0, "W": float("nan"),
                "p": 1.0, "rank_biserial": 0.0}
    res = sps.wilcoxon(x, y, zero_method="wilcox", alternative="two-sided")
    ranks = sps.rankdata(np.abs(nz))
    wp, wm = ranks[nz > 0].sum(), ranks[nz < 0].sum()
    return {"n": int(len(d)), "median_diff": float(np.median(d)), "mean_diff": float(d.mean()),
            "W": float(res.statistic), "p": float(res.pvalue),
            "rank_biserial": float((wp - wm) / (wp + wm))}


def holm(pvals: Dict[str, float]) -> Dict[str, float]:
    """Holm-Bonferroni düzeltilmiş p-değerleri (aile bazlı hata kontrolü)."""
    keys = sorted(pvals, key=lambda k: pvals[k])
    m = len(keys)
    adj, running = {}, 0.0
    for i, k in enumerate(keys):
        running = max(running, min(1.0, (m - i) * pvals[k]))
        adj[k] = running
    return adj


def hier_bootstrap_ci(mat: np.ndarray, n_boot: int = 10000, level: float = 0.95, seed: int = 0) -> List[float]:
    """mat: (alan, tohum). İki aşamalı bootstrap: önce alanlar, sonra her alanda tohumlar."""
    rng = np.random.default_rng(seed)
    D, S = mat.shape
    di = rng.integers(0, D, size=(n_boot, D))
    si = rng.integers(0, S, size=(n_boot, D, S))
    boots = mat[di[..., None], si].mean(axis=(1, 2))
    a = (1 - level) / 2
    return [float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))]


def spearman(x, y) -> float:
    return float(sps.spearmanr(x, y).statistic)
