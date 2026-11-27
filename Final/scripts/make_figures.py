"""Makale/sunum figürleri — tümü results/ altındaki gerçek çıktılardan çizilir."""

import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttamon import config as C  # noqa: E402
from ttamon.data import DOMAINS, cached_domain  # noqa: E402
from ttamon.metrics import softmax_np  # noqa: E402
from ttamon.paths import FIG_DIR, LOGS_DIR, RAW_DIR, REPO_ROOT, RESULTS_DIR, TABLES_DIR  # noqa: E402

# Doğrulanmış varsayılan kategorik palet (dataviz referansı, açık mod) — sabit sıra
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
MCOL = {"Source": SERIES[0], "BN-adapt": SERIES[1], "TENT": SERIES[2]}
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9.5, "axes.titlesize": 10.5,
    "axes.titleweight": "bold", "legend.frameon": False, "lines.linewidth": 2.0,
})
DOMAIN_TR = {"clean": "temiz", "cifar10_1": "CIFAR-10.1"}


def dlabel(d):
    return DOMAIN_TR.get(d, d.replace("_", " "))


def save(fig, name):
    p = os.path.join(FIG_DIR, name)
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("şekil:", p)


def fig_training():
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    for i, s in enumerate(C.MODEL_SEEDS):
        cands = [os.path.join(REPO_ROOT, "results", "logs", f"cifar10_training_curves_seed{s}.json"),
                 os.path.join(LOGS_DIR, f"train_curves_seed{s}.json")]
        p = next((c for c in cands if os.path.exists(c)), None)
        if p is None:
            continue
        h = json.load(open(p))
        ax.plot(h["epochs"], h["val_acc"], color=SERIES[i], label=f"tohum {s}")
    ax.set_xlabel("epok")
    ax.set_ylabel("kaynak doğrulama doğruluğu (%)")
    ax.set_ylim(40, 100)
    ax.legend(ncol=5, loc="lower right", fontsize=8.5)
    ax.set_title("Kaynak ResNet-18 modelleri: doğrulama eğrileri (5 bağımsız tohum)", loc="left")
    save(fig, "fig1_training_curves.png")


def fig_per_domain(df):
    v = df[(df.cond == "iid64") & (df.stream_seed == C.MAIN_STREAM_SEED) & df.variant.isin(C.MAIN_METHODS)]
    g = v.groupby(["domain", "variant"])["acc"].agg(["mean", "std"]).reset_index()
    order = DOMAINS[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    y = np.arange(len(order))
    for k, m in enumerate(C.MAIN_METHODS):
        gm = g[g.variant == m].set_index("domain").loc[order]
        ax.errorbar(gm["mean"], y + (k - 1) * 0.22, xerr=gm["std"], fmt="o", ms=5, color=MCOL[m],
                    elinewidth=1.2, capsize=0, label=m, markeredgecolor=SURF, markeredgewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([dlabel(d) for d in order])
    ax.set_xlabel(f"doğruluk (%) — ortalama ± std, {NSEED} model")
    ax.legend(loc="lower left", ncol=3)
    ax.set_title("iid64: alan bazlı doğruluk", loc="left")
    save(fig, "fig2_per_domain_accuracy.png")


def fig_reliability():
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2), sharey=True)
    edges = np.linspace(0, 1, 16)
    for ax, m in zip(axes, C.MAIN_METHODS):
        confs, cors = [], []
        for p in glob.glob(os.path.join(RAW_DIR, "iid64", m, f"*_st{C.MAIN_STREAM_SEED}.npz")):
            dom = os.path.basename(p).rsplit("_", 2)[0]
            if dom == "clean":
                continue
            z = np.load(p)
            pr = softmax_np(z["logits"].astype(np.float32))
            y = cached_domain(dom)[1][z["idx"]]
            confs.append(pr.max(1))
            cors.append(pr.argmax(1) == y)
        if not confs:
            continue
        conf, cor = np.concatenate(confs), np.concatenate(cors)
        mids, accs, fr = [], [], []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mk = (conf > lo) & (conf <= hi)
            if mk.sum() > 50:
                mids.append(conf[mk].mean())
                accs.append(cor[mk].mean())
                fr.append(mk.mean())
        ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls="--")
        ax.plot(mids, accs, "-o", color=MCOL[m], ms=4)
        ax.set_title(f"{m}\nort. güven {conf.mean()*100:.1f} · doğruluk {cor.mean()*100:.1f}", fontsize=9.5)
        ax.set_xlabel("güven")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("gerçek doğruluk")
    fig.suptitle(f"Güvenilirlik diyagramları — iid64, 16 kaymalı alan (örnek ağırlıklı havuz), {NSEED} model", x=0.01, ha="left",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig3_reliability.png")


def fig_est_vs_true():
    p = os.path.join(TABLES_DIR, "domain_level_estimates.csv")
    if not os.path.exists(p):
        return
    a = pd.read_csv(p)
    a = a[a.cond == "iid64"]
    ests = [("est_DoC_src", "DoC (kaynak ref.)"), ("est_ATC-MC_src", "ATC-MC (kaynak ref.)"), ("est_GDE", "GDE (uyuşma)")]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.3), sharex=True, sharey=True)
    for ax, (col, name) in zip(axes, ests):
        ax.plot([0, 100], [0, 100], color=INK2, lw=1, ls="--")
        for m in C.MAIN_METHODS:
            s = a[a.variant == m]
            ax.scatter(s["acc"], s[col], s=22, color=MCOL[m], label=m, edgecolor=SURF, linewidth=0.8)
        mae = (a[col] - a["acc"]).abs().mean()
        ax.set_title(f"{name}\nMAE (tüm yöntemler) = {mae:.2f} pp", fontsize=9.5)
        ax.set_xlabel("gerçek doğruluk (%)")
        ax.set_xlim(15, 100)
        ax.set_ylim(15, 100)
    axes[0].set_ylabel("etiketsiz tahmin (%)")
    axes[0].legend(loc="upper left", fontsize=8.5)
    fig.suptitle(f"Tahmin vs gerçek doğruluk — iid64, alan düzeyi ({NSEED} model ortalaması)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig4_estimate_vs_true.png")


def fig_mae_heatmap(S):
    rq2 = S["rq2"]
    cols = [(c, m) for c in C.CONDITIONS for m in C.MAIN_METHODS]
    names = list(rq2["iid64"]["TENT"].keys())
    M = np.array([[rq2[c][m][n]["mae"]["mean"] for (c, m) in cols] for n in names])
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    vmax = np.nanpercentile(M, 95)
    im = ax.imshow(np.minimum(M, vmax), cmap=cmap, aspect="auto", vmin=0, vmax=vmax)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center", fontsize=7.5,
                    color="white" if M[i, j] > vmax * 0.6 else INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{c}\n{m}" for c, m in cols], fontsize=7.5)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.03)
    cb.set_label("MAE (pp)")
    ax.set_title(f"Etiketsiz tahmin hatası (MAE, 16 kaymalı alan, {len(S['seeds_present'])} model ort.)", loc="left")
    save(fig, "fig5_mae_heatmap.png")


def fig_gate(S):
    pol = S["gates"]["policies"]
    keep = [("always_Source", "hep Source"), ("always_BN-adapt", "hep BN-adapt"), ("always_TENT", "hep TENT"),
            ("gate_GDE", "kapı: GDE"), ("gate_ATC-MC_src", "kapı: ATC-MC (src)"), ("gate_DoC_matched", "kapı: DoC (eşl.)")]
    keep = [k for k in keep if k[0] in pol]
    fig, axes = plt.subplots(1, len(C.CONDITIONS), figsize=(8.6, 3.0), sharey=True)
    for ax, cond in zip(axes, C.CONDITIONS):
        ax.set_axisbelow(True)
        vals = [pol[k][cond]["mean"] for k, _ in keep]
        errs = [pol[k][cond]["std"] for k, _ in keep]
        y = np.arange(len(keep))
        ax.barh(y, vals, xerr=errs, color=SERIES[0], height=0.6, error_kw={"elinewidth": 1, "ecolor": INK2})
        ax.grid(axis="y", visible=False)
        for yi, v, e in zip(y, vals, errs):
            ax.text(v + e + max(vals) * 0.03 + 0.05, yi, f"{v:.2f}", va="center", fontsize=8, color=INK2)
        ax.set_xlim(0, max(v + e for v, e in zip(vals, errs)) * 1.18 + 0.5)
        ax.set_yticks(y)
        ax.set_yticklabels([n for _, n in keep])
        ax.invert_yaxis()
        ax.set_title(cond)
        ax.set_xlabel("pişmanlık (pp)")
    fig.suptitle("Uyarlama kapısı: oracle'a göre pişmanlık (düşük = iyi)", x=0.01, ha="left", fontweight="bold",
                 fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig6_gate_regret.png")


def fig_ablation(S):
    ab = S["ablations"]
    vs = [v for v in ["TENT-lr1e-4", "TENT", "TENT-lr1e-2", "TENT-after"] if v in ab]
    lab = {"TENT-lr1e-4": "lr 1e-4", "TENT": "lr 1e-3\n(ana)", "TENT-lr1e-2": "lr 1e-2", "TENT-after": "lr 1e-3\ngünc.-sonrası"}
    fig, axes = plt.subplots(1, 3, figsize=(9.4, 3.0))
    for ax, (key, name) in zip(axes, [("acc", "doğruluk (%)"), ("overconf", "aşırı güven (pp)"),
                                      ("aerr_GDE", "MAE (pp)")]):
        x = np.arange(len(vs))
        if key == "aerr_GDE":
            w = 0.38
            for k, (kk, nn, col) in enumerate([("aerr_ATC-MC_src", "ATC-MC (src)", SERIES[0]),
                                               ("aerr_GDE", "GDE", SERIES[1])]):
                ax.bar(x + (k - 0.5) * w, [ab[v][kk]["mean"] for v in vs], w * 0.92,
                       yerr=[ab[v][kk]["std"] for v in vs], color=col, label=nn,
                       error_kw={"elinewidth": 1, "ecolor": INK2})
            ax.legend(fontsize=8)
        else:
            ax.bar(x, [ab[v][key]["mean"] for v in vs], 0.6, yerr=[ab[v][key]["std"] for v in vs],
                   color=SERIES[2], error_kw={"elinewidth": 1, "ecolor": INK2})
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x)
        ax.set_xticklabels([lab[v] for v in vs], fontsize=8)
        ax.set_title(name)
    fig.suptitle("TENT ablasyonları — iid64, 16 kaymalı alan", x=0.01, ha="left", fontweight="bold", fontsize=10.5)
    fig.tight_layout()
    save(fig, "fig7_tent_ablation.png")


NSEED = 5


def main():
    global NSEED
    sp0 = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.exists(sp0):
        NSEED = len(json.load(open(sp0, encoding="utf-8"))["seeds_present"])
    fig_training()
    p = os.path.join(RESULTS_DIR, "all_runs.csv")
    if os.path.exists(p):
        df = pd.read_csv(p)
        fig_per_domain(df)
        fig_reliability()
        fig_est_vs_true()
    sp = os.path.join(RESULTS_DIR, "summary.json")
    if os.path.exists(sp):
        S = json.load(open(sp, encoding="utf-8"))
        fig_mae_heatmap(S)
        fig_gate(S)
        fig_ablation(S)


if __name__ == "__main__":
    main()
