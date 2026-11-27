"""
Denetim (AUDIT) doğrulama betiği — orijinal depodaki iddiaları gerçek çalıştırmalarla sınar.

Kontroller:
  V1  Checkpoint SHA256'ları (dondurulmuş 42/43/44) + yeni modeller (45/46) için hash kaydı
  V2  Temiz test (10k) ve kaynak doğrulama (5k) doğrulukları; checkpoint içindeki en iyi epok
  V3  Faz-1 sonuçlarının ORİJİNAL kod yolu (src/evaluate_phase1.evaluate_single_run) ile kısmi yeniden üretimi
  V4  Veri bütünlüğü: CIFAR-10-C etiket hizası, görüntü hizası (korelasyon), Gauss gürültüsü σ kontrolü
  V5  Rapor sayılarının JSON'dan yeniden hesaplanması (ortalama, aile ortalamaları, "GA"ların doğası)
  V6  Orijinal benchmark_e2e.py'deki TENT optimizer hatasının gösterimi
  V7  ResNet-18 parametre sayısı

Çıktı: Final/results/logs/audit_verification.json
Kullanım (depo kökünden bağımsız): python scripts/verify_audit.py [--skip-phase1]
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
import torch

FINAL = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(FINAL)
sys.path.insert(0, FINAL)
from ttamon.data import CORRUPTION_FAMILIES, load_cifar10_test, load_domain, load_val_split, to_gpu_normalized  # noqa
from ttamon.models import get_cifar_resnet18, load_source_model  # noqa
from ttamon.paths import LOGS_DIR, ckpt_path  # noqa

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT = {}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def eval_acc(model, x, y, bs=500):
    xg = to_gpu_normalized(x, DEV)
    pred = torch.cat([model(xg[i:i + bs]).argmax(1).cpu() for i in range(0, len(y), bs)]).numpy()
    return float((pred == y).mean() * 100)


def v1_v2():
    frozen = json.load(open(os.path.join(ROOT, "results/logs/checkpoint_hashes.json"), encoding="utf-8"))
    xt, yt = load_cifar10_test()
    rows, new_hashes = {}, {}
    for s in [42, 43, 44, 45, 46]:
        p = ckpt_path(s)
        if not os.path.exists(p):
            rows[s] = {"exists": False}
            continue
        h = sha(p)
        ck = torch.load(p, map_location="cpu", weights_only=False)
        m = load_source_model(p, DEV)
        xv, yv = load_val_split(s)
        r = {"sha256": h, "best_epoch_in_ckpt": int(ck.get("epoch", -1)),
             "val_acc_in_ckpt": float(ck.get("val_acc", float("nan"))),
             "clean_test_acc_recomputed": round(eval_acc(m, xt, yt), 2),
             "val_acc_recomputed": round(eval_acc(m, xv, yv), 2)}
        key = f"seed{s}"
        if key in frozen:
            r["sha256_matches_frozen"] = (h == frozen[key]["sha256"])
            r["claimed_clean_test_acc"] = frozen[key]["clean_test_acc"]
            r["claimed_val_acc"] = frozen[key]["val_acc"]
            r["clean_acc_matches_claim"] = abs(r["clean_test_acc_recomputed"] - frozen[key]["clean_test_acc"]) < 0.005
        new_hashes[key] = {"path": os.path.relpath(p, ROOT).replace("\\", "/"), "sha256": h,
                           "val_acc": r["val_acc_recomputed"], "clean_test_acc": r["clean_test_acc_recomputed"],
                           "best_epoch": r["best_epoch_in_ckpt"]}
        rows[s] = r
        print(f"[V1/V2] seed {s}: {r}")
    OUT["V1_V2_checkpoints"] = rows
    with open(os.path.join(LOGS_DIR, "checkpoint_hashes_final.json"), "w", encoding="utf-8") as f:
        json.dump(new_hashes, f, indent=2)


def v3_phase1_repro():
    """Orijinal kod yolu ile seed 42 ve 43 için 3 bozulma x 3 yöntem yeniden koşturulur."""
    cwd = os.getcwd()
    os.chdir(ROOT)
    sys.path.insert(0, ROOT)
    try:
        from src.evaluate_phase1 import evaluate_single_run  # orijinal kod
        rep = json.load(open("results/logs/phase1_cifar10_c_results.json", encoding="utf-8"))
        reported = {(r["method"], r["corruption"], r["seed"]): r for r in rep["raw_runs"]}
        rows = []
        for seed in [42, 43]:
            for corr in ["gaussian_noise", "contrast", "jpeg_compression"]:
                for meth in ["Source", "BN-adapt", "TENT"]:
                    torch.manual_seed(0)
                    r = evaluate_single_run(meth, corr, seed, f"src/models/checkpoints/resnet18_cifar10_seed{seed}.pth",
                                            batch_size=64, device=DEV)
                    rr = reported[(meth, corr, seed)]
                    rows.append({"seed": seed, "corruption": corr, "method": meth,
                                 "reported_acc": rr["accuracy"], "reproduced_acc": r["accuracy"],
                                 "abs_diff": round(abs(rr["accuracy"] - r["accuracy"]), 2),
                                 "reported_ece": rr["ece"], "reproduced_ece": r["ece"]})
                    print(f"[V3] {rows[-1]}")
        OUT["V3_phase1_reproduction"] = {"rows": rows, "max_abs_diff_acc": max(r["abs_diff"] for r in rows)}
    finally:
        os.chdir(cwd)


def v4_data():
    xt, yt = load_cifar10_test()
    res = {}
    _, y5 = load_domain("gaussian_noise")
    res["cifar10c_labels_equal_cifar10_test_labels"] = bool((y5 == yt).all())
    for c in ["gaussian_noise", "brightness", "contrast", "jpeg_compression"]:
        x, _ = load_domain(c)
        a = x.reshape(len(x), -1).astype(np.float32)
        b = xt.reshape(len(xt), -1).astype(np.float32)
        a -= a.mean(1, keepdims=True)
        b -= b.mean(1, keepdims=True)
        r = (a * b).sum(1) / np.sqrt((a * a).sum(1) * (b * b).sum(1) + 1e-9)
        res[f"{c}_mean_pixel_corr_with_clean"] = round(float(r.mean()), 4)
    xg, _ = load_domain("gaussian_noise")
    resid = (xg.astype(np.float32) - xt.astype(np.float32)) / 255.0
    inner = (xt > 40) & (xt < 215)   # kırpılmadan (clipping) etkilenmeyen pikseller
    res["gaussian_noise_residual_std_unclipped_pixels"] = round(float(resid[inner].std()), 4)
    res["note"] = ("CIFAR-10-C şiddet 5 Gauss gürültüsü için Hendrycks & Dietterich kodunda σ=0.10'dur; "
                   "kalıntı std'nin ≈0.10 olması dilimin şiddet 5 olduğuyla tutarlıdır (resmi Zenodo "
                   "sağlama toplamı ile birebir doğrulama: DOĞRULANAMADI).")
    x1, y1 = load_domain("cifar10_1")
    res["cifar10_1_shape"] = list(x1.shape)
    res["cifar10_1_class_counts"] = np.bincount(y1).tolist()
    OUT["V4_data"] = res
    print(f"[V4] {res}")


def v5_report_numbers():
    rep = json.load(open(os.path.join(ROOT, "results/logs/phase1_cifar10_c_results.json"), encoding="utf-8"))
    sbc = rep["summary_by_corruption"]
    issues = []
    ci_is_range = 0
    for c, v in sbc.items():
        d = v["paired_delta_tent_minus_bn"]
        if [min(d["seed_deltas"]), max(d["seed_deltas"])] == d["ci_95_delta"]:
            ci_is_range += 1
        for m in ["Source", "BN-adapt", "TENT"]:
            mu = round(float(np.mean(v[m]["seed_accs"])), 2)
            if abs(mu - v[m]["mean_acc"]) > 0.011:
                issues.append(f"{c}/{m}: mean mismatch {mu} vs {v[m]['mean_acc']}")
    # gürültü ailesi için raporda verilen "[7.09, 9.21]" aralığı
    noise = CORRUPTION_FAMILIES["noise"]
    lows = [sbc[c]["paired_delta_tent_minus_bn"]["ci_95_delta"][0] for c in noise]
    highs = [sbc[c]["paired_delta_tent_minus_bn"]["ci_95_delta"][1] for c in noise]
    # doğru hesap: seed başına aile ortalaması farkı, t-GA (n=3)
    from scipy import stats as sps
    per_seed = np.mean([sbc[c]["paired_delta_tent_minus_bn"]["seed_deltas"] for c in noise], axis=0)
    h = sps.t.ppf(0.975, 2) * per_seed.std(ddof=1) / np.sqrt(3)
    OUT["V5_report_numbers"] = {
        "mean_recompute_issues": issues,
        "n_corruptions_where_reported_95CI_equals_min_max_of_3_seeds": ci_is_range,
        "noise_family_reported_interval": [7.09, 9.21],
        "noise_family_interval_is_min_of_lows_and_max_of_highs": [min(lows), max(highs)],
        "noise_family_per_seed_mean_delta": [round(float(v), 3) for v in per_seed],
        "noise_family_t_CI95_n3": [round(float(per_seed.mean() - h), 2), round(float(per_seed.mean() + h), 2)],
        "overall_mean_delta": rep["overall_summary"]["paired_delta_tent_minus_bn"]["mean_delta"],
        "phase1_total_minutes": rep["metadata"]["total_duration_minutes"],
    }
    print(f"[V5] {OUT['V5_report_numbers']}")


def v6_benchmark_bug():
    sys.path.insert(0, ROOT)
    import torch.optim as optim
    from src.methods.tent import TentAdapter
    from src.methods.bn_adapt import BlendedBatchNorm2d
    m = get_cifar_resnet18(10)
    opt = optim.Adam(m.parameters(), lr=1e-3)          # benchmark_e2e.py'deki gibi (dönüşümden ÖNCE)
    ad = TentAdapter(m, opt, steps=1, blend_lambda=0.9)
    bn = [x for x in ad.model.modules() if isinstance(x, BlendedBatchNorm2d)][0]
    w0 = bn.weight.detach().clone()
    ad.adapt_and_predict(torch.randn(8, 3, 32, 32))
    opt_ids = {id(p) for g in opt.param_groups for p in g["params"]}
    OUT["V6_benchmark_tent_bug"] = {
        "bn_affine_params_in_optimizer": sum(id(p) in opt_ids for p in [bn.weight, bn.bias]),
        "bn_weight_changed_after_step": bool(not torch.equal(w0, bn.weight.detach())),
        "conclusion": "Optimizer, BN dönüşümünden önceki (artık modelde olmayan) parametreleri tutuyor; "
                      "benchmark'ta TENT hiçbir parametreyi güncellemiyor (yalnızca ileri+geri geçiş süresi ölçülüyor).",
    }
    print(f"[V6] {OUT['V6_benchmark_tent_bug']}")


def v7_params():
    OUT["V7_resnet18_params"] = int(sum(p.numel() for p in get_cifar_resnet18(10).parameters()))
    print(f"[V7] params={OUT['V7_resnet18_params']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-phase1", action="store_true")
    a = ap.parse_args()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    v7_params()
    v6_benchmark_bug()
    v5_report_numbers()
    v4_data()
    v1_v2()
    if not a.skip_phase1:
        v3_phase1_repro()
    path = os.path.join(LOGS_DIR, "audit_verification.json")
    prev = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
    prev.update(OUT)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(prev, f, indent=2, ensure_ascii=False)
    print(f"Kaydedildi -> {path}")
