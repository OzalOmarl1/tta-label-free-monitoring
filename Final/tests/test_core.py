"""Final paketi birim testleri: yöntem bütünlüğü, akışlar, tahmin ediciler, istatistik, termal koruyucu."""

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttamon import thermal_guard as tg  # noqa: E402
from ttamon.data import make_stream  # noqa: E402
from ttamon.estimators import Reference, estimate_all, fit_temperature  # noqa: E402
from ttamon.methods import BlendedBatchNorm2d, BNAdaptAdapter, SourceAdapter, TentAdapter  # noqa: E402
from ttamon.metrics import ece, softmax_np  # noqa: E402
from ttamon.models import get_cifar_resnet18  # noqa: E402
from ttamon.stats import holm, wilcoxon_paired  # noqa: E402


def _model():
    torch.manual_seed(0)
    m = get_cifar_resnet18(10)
    # BN istatistiklerini sıfırdan farklı yap
    for mod in m.modules():
        if isinstance(mod, torch.nn.BatchNorm2d):
            mod.running_mean.uniform_(-0.1, 0.1)
            mod.running_var.uniform_(0.5, 1.5)
    return m.eval()


def test_tent_official_first_batch_equals_bn_adapt():
    """Resmi Tent protokolü: ilk partinin tahmini güncellemeden önceki ileri geçiştir => BN-adapt ile özdeş."""
    x = torch.randn(8, 3, 32, 32)
    bn = BNAdaptAdapter(_model())(x)
    tent = TentAdapter(_model(), lr=1e-2, predict="official")(x)
    assert torch.allclose(bn, tent, atol=1e-5)
    after = TentAdapter(_model(), lr=1e-2, predict="after_update")(x)
    assert not torch.allclose(bn, after, atol=1e-5)


def test_tent_updates_only_bn_affine():
    m = _model()
    conv0 = m.conv1.weight.detach().clone()
    fc0 = m.fc.weight.detach().clone()
    ad = TentAdapter(m, lr=1e-2)
    bn = [x for x in ad.model.modules() if isinstance(x, BlendedBatchNorm2d)][0]
    w0 = bn.weight.detach().clone()
    ad(torch.randn(8, 3, 32, 32))
    assert torch.equal(ad.model.conv1.weight, conv0) and torch.equal(ad.model.fc.weight, fc0)
    assert not torch.equal(bn.weight.detach(), w0)
    assert ad.model.conv1.weight.grad is None


def test_bn_adapt_and_source_no_grad():
    x = torch.randn(4, 3, 32, 32)
    for ad in (SourceAdapter(_model()), BNAdaptAdapter(_model())):
        out = ad(x)
        assert out.shape == (4, 10) and not out.requires_grad
        assert all(p.grad is None for p in ad.model.parameters())


def test_streams_are_permutations():
    y = np.repeat(np.arange(10), 30)
    for kind in ("iid", "sorted"):
        o = make_stream(y, kind, 3)
        assert sorted(o.tolist()) == list(range(len(y)))
    o = make_stream(y, "sorted", 3)
    changes = int((np.diff(y[o]) != 0).sum())
    assert changes == 9  # 10 sınıf bloğu
    assert np.array_equal(make_stream(y, "iid", 5), make_stream(y, "iid", 5))


def test_estimators_on_reference_distribution():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 10, 4000)
    logits = rng.normal(0, 1, (4000, 10))
    logits[np.arange(4000), y] += 2.5
    ref = Reference(logits, y)
    est = estimate_all(logits, ref)
    acc = 100 * (logits.argmax(1) == y).mean()
    assert abs(est["DoC"] - acc) < 1e-6           # DoC referansta tam doğru
    assert abs(est["ATC-MC"] - acc) < 0.1          # ATC eşik tanımı gereği
    assert abs(est["ATC-NE"] - acc) < 0.1


def test_temperature_recovery():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 2, (20000, 10))
    p = softmax_np(base)
    y = np.array([rng.choice(10, p=pi) for pi in p])
    T = fit_temperature(base * 3.0, y)            # logitler 3 kat ölçeklendi -> T≈3
    assert abs(T - 3.0) < 0.25


def test_ece_perfect_calibration_near_zero():
    rng = np.random.default_rng(2)
    p = softmax_np(rng.normal(0, 2, (50000, 10)))
    y = np.array([rng.choice(10, p=pi) for pi in p[:20000]])
    assert ece(p[:20000], y) < 1.5


def test_holm_and_wilcoxon():
    adj = holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adj["a"] == pytest.approx(0.03) and adj["c"] == pytest.approx(0.06) and adj["b"] == pytest.approx(0.06)
    r = wilcoxon_paired(np.arange(1, 17) + 1.0, np.arange(1, 17))
    assert r["p"] < 0.001 and r["rank_biserial"] == pytest.approx(1.0)


def test_thermal_guard_pauses_and_resumes(monkeypatch):
    seq = iter([90.0, 90.0, 60.0, 60.0, 60.0, 60.0])
    monkeypatch.setattr(tg, "read_gpu_temp", lambda: next(seq, 60.0))
    monkeypatch.setattr(tg, "read_cpu_temp", lambda: (50.0, "test"))
    monkeypatch.setattr(tg, "read_gpu_util", lambda: None)
    g = tg.ThermalGuard(poll_seconds=0.01, verbose=False)
    g.poll_once()
    assert g.level == "hot"
    g.throttle()                                   # soğuyana kadar bekler (iş parçacığı yok -> kendisi okur)
    assert g.n_pauses == 1 and g.level == "ok"
    assert g.recommend_batch_size(64) == 64
    g.level = "hot"
    assert g.recommend_batch_size(64) == 16


def test_fused_bn_equals_manual():
    torch.manual_seed(3)
    bn = torch.nn.BatchNorm2d(5)
    bn.weight.data.uniform_(0.5, 1.5)
    bn.bias.data.uniform_(-0.5, 0.5)
    m = BlendedBatchNorm2d(bn)
    x = torch.randn(16, 5, 8, 8) * 3 + 1
    mu = x.mean(dim=[0, 2, 3], keepdim=True)
    var = x.var(dim=[0, 2, 3], unbiased=False, keepdim=True)
    ref = (x - mu) / torch.sqrt(var + m.eps) * m.weight.view(1, 5, 1, 1) + m.bias.view(1, 5, 1, 1)
    assert torch.allclose(m(x), ref, atol=1e-5)
