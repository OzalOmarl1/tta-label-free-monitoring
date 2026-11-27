"""
Tüm TTA koşumlarını yürütür ve her koşum için logitleri (float16) kaydeder.

Çıktı düzeni (Final/results/raw/):
  {koşul}/{varyant}/{alan}_s{model_tohumu}_st{akış_tohumu}.npz   -> logits, idx (akış sırası)
  val/N{batch}/{varyant}/s{model_tohumu}.npz                      -> kaynak doğrulama kümesi logitleri

Her koşum: model checkpoint'ten sıfırdan yüklenir (alan başına episodik sıfırlama), akış
sırasıyla N'lik partiler halinde işlenir; etiketler adaptasyonda KULLANILMAZ.
Termal koruyucu her partiden önce çağrılır (yalnızca uyku/duraklatma; N değişmez).
Var olan dosyalar atlanır (kaldığı yerden devam).

Kullanım:
  python scripts/collect_predictions.py --plan val main ablation stream
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttamon import config as C  # noqa: E402
from ttamon.data import DOMAINS, cached_domain, load_val_split, make_stream, to_gpu_normalized  # noqa: E402
from ttamon.methods import build_adapter  # noqa: E402
from ttamon.models import load_source_model  # noqa: E402
from ttamon.paths import LOGS_DIR, RAW_DIR, ckpt_path  # noqa: E402
from ttamon.thermal_guard import ThermalGuard  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def run_stream(variant: str, seed: int, x_gpu: torch.Tensor, order: np.ndarray, bs: int,
               guard: ThermalGuard) -> np.ndarray:
    base, kw = C.VARIANTS[variant]
    torch.manual_seed(seed)
    model = load_source_model(ckpt_path(seed), DEVICE)
    adapter = build_adapter(base, model, **kw)
    order_t = torch.from_numpy(order).to(DEVICE)
    outs = []
    for i in range(0, len(order), bs):
        guard.throttle()
        xb = x_gpu.index_select(0, order_t[i:i + bs])
        outs.append(adapter(xb).float().cpu())
    return torch.cat(outs).numpy()


def save(path: str, logits: np.ndarray, order: np.ndarray, meta: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp.npz"
    np.savez_compressed(tmp, logits=logits.astype(np.float16), idx=order.astype(np.int32),
                        meta=json.dumps(meta))
    os.replace(tmp, path)


def jobs_for(plan: str):
    """(tür, koşul, varyant, alan, model_tohumu, akış_tohumu) üreteci."""
    if plan == "val":
        variants = C.MAIN_METHODS + C.ABLATION_VARIANTS
        for bs in sorted({v[1] for v in C.CONDITIONS.values()}):
            for v in variants:
                if bs != 64 and v in C.ABLATION_VARIANTS:
                    continue
                for s in C.MODEL_SEEDS:
                    yield ("val", f"N{bs}", v, "val", s, C.MAIN_STREAM_SEED)
    elif plan == "main":
        for cond in C.CONDITIONS:
            for d in DOMAINS:
                for v in C.MAIN_METHODS:
                    for s in C.MODEL_SEEDS:
                        yield ("test", cond, v, d, s, C.MAIN_STREAM_SEED)
    elif plan == "ablation":
        for d in DOMAINS:
            for v in C.ABLATION_VARIANTS:
                for s in C.MODEL_SEEDS:
                    yield ("test", "iid64", v, d, s, C.MAIN_STREAM_SEED)
    elif plan == "stream":
        for d in DOMAINS:
            for v in C.STREAM_METHODS:
                for st in C.STREAM_ABLATION_SEEDS:
                    yield ("test", "iid64", v, d, C.STREAM_ABLATION_MODEL, st)
    else:
        raise ValueError(plan)


def out_path(kind, cond, variant, domain, seed, st):
    if kind == "val":
        return os.path.join(RAW_DIR, "val", cond, variant, f"s{seed}.npz")
    return os.path.join(RAW_DIR, cond, variant, f"{domain}_s{seed}_st{st}.npz")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", nargs="+", default=["val", "main", "ablation", "stream"])
    ap.add_argument("--seeds", nargs="*", type=int, default=None, help="yalnızca bu model tohumları")
    ap.add_argument("--domains", nargs="*", default=None, help="yalnızca bu alanlar (val işleri etkilenmez)")
    ap.add_argument("--conds", nargs="*", default=None, help="yalnızca bu koşullar")
    args = ap.parse_args()

    jobs = [j for p in args.plan for j in jobs_for(p)]
    if args.seeds:
        jobs = [j for j in jobs if j[4] in args.seeds]
    if args.domains:
        jobs = [j for j in jobs if j[0] == "val" or j[3] in args.domains]
    if args.conds:
        jobs = [j for j in jobs if j[0] == "val" or j[1] in args.conds]
    todo = [j for j in jobs if not os.path.exists(out_path(*j))]
    print(f"Toplam iş: {len(jobs)} | yapılacak: {len(todo)} | cihaz: {DEVICE}", flush=True)
    if not todo:
        return

    guard = ThermalGuard(log_csv=os.path.join(LOGS_DIR, "thermal_collect.csv")).start()
    runlog = open(os.path.join(LOGS_DIR, "collect_runs.jsonl"), "a", encoding="utf-8")
    gpu_cache = {}
    val_cache = {}
    t_all = time.time()
    try:
        for k, (kind, cond, variant, domain, seed, st) in enumerate(todo, 1):
            if kind == "val":
                bs = int(cond[1:])
                if seed not in val_cache:
                    xv, yv = load_val_split(seed)
                    val_cache.clear()
                    val_cache[seed] = (to_gpu_normalized(xv, DEVICE), yv)
                x_gpu, y = val_cache[seed]
                order = make_stream(y, "iid", st)
            else:
                kind_stream, bs = C.CONDITIONS[cond]
                if domain not in gpu_cache:
                    gpu_cache.clear()
                    if DEVICE == "cuda":
                        torch.cuda.empty_cache()
                    xd, yd = cached_domain(domain)
                    gpu_cache[domain] = (to_gpu_normalized(xd, DEVICE), yd)
                x_gpu, y = gpu_cache[domain]
                order = make_stream(y, kind_stream, st)
            t0 = time.time()
            logits = run_stream(variant, seed, x_gpu, order, bs, guard)
            dt = time.time() - t0
            acc = float((logits.argmax(1) == y[order]).mean() * 100)
            meta = {"kind": kind, "cond": cond, "variant": variant, "domain": domain, "seed": seed,
                    "stream_seed": st, "batch_size": bs, "seconds": round(dt, 2), "acc": round(acc, 2),
                    "torch": torch.__version__, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
            save(out_path(kind, cond, variant, domain, seed, st), logits, order, meta)
            runlog.write(json.dumps(meta) + "\n")
            runlog.flush()
            print(f"[{k:4d}/{len(todo)}] {cond:8s} {variant:12s} {domain:18s} s{seed} st{st} "
                  f"acc={acc:6.2f} {dt:5.1f}s | GPU {guard.gpu_temp} CPU {guard.cpu_temp} {guard.level}",
                  flush=True)
    finally:
        guard.stop()
        runlog.close()
        print(f"Bitti. Süre {(time.time()-t_all)/60:.1f} dk | termal özet: {guard.summary()}", flush=True)
        with open(os.path.join(LOGS_DIR, "thermal_collect_summary.json"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "plans": args.plan,
                                **guard.summary()}) + "\n")


if __name__ == "__main__":
    main()
