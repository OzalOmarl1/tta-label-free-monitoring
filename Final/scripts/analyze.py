"""
Tüm analizler: metrikler, etiketsiz tahminler, hipotez testleri, ablasyonlar, hata analizi.

Girdi : Final/results/raw/**.npz  (collect_predictions.py)
Çıktı : Final/results/all_runs.csv            (her koşum için tüm metrik + tahminler)
        Final/results/summary.json            (makale/sunum üreticilerinin okuduğu tüm sayılar)
        Final/results/tables/*.md|csv         (tablolar)
Rakamların TAMAMI bu betik tarafından ham logitlerden hesaplanır; elle girilen sayı yoktur.
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ttamon import config as C  # noqa: E402
from ttamon.data import CIFAR10_CLASSES, DOMAINS, cached_domain, domain_family, load_val_split  # noqa: E402
from ttamon.estimators import CONF_ESTIMATORS, Reference, estimate_all  # noqa: E402
from ttamon.metrics import all_metrics, softmax_np  # noqa: E402
from ttamon.paths import LOGS_DIR, RAW_DIR, RESULTS_DIR, TABLES_DIR  # noqa: E402
from ttamon.stats import hier_bootstrap_ci, holm, mean_std, spearman, t_ci, wilcoxon_paired  # noqa: E402

SHIFTED = [d for d in DOMAINS if d != "clean"]
METHODS = C.MAIN_METHODS
REFS = ["src", "matched"]
SUMMARY = {}


def load_npz(p):
    z = np.load(p, allow_pickle=False)
    return z["logits"].astype(np.float32), z["idx"], json.loads(str(z["meta"]))


# ---------------------------------------------------------------------------
# 1) Referanslar (doğrulama kümesi)
# ---------------------------------------------------------------------------
def build_refs():
    refs, val_acc = {}, {}
    ylab = {}
    for p in glob.glob(os.path.join(RAW_DIR, "val", "*", "*", "*.npz")):
        logits, idx, m = load_npz(p)
        s = m["seed"]
        if s not in ylab:
            ylab[s] = load_val_split(s)[1]
        y = ylab[s][idx]
        refs[(m["cond"], m["variant"], s)] = Reference(logits, y)
        val_acc[(m["cond"], m["variant"], s)] = float((logits.argmax(1) == y).mean() * 100)
    return refs, val_acc


def ref_for(refs, kind, cond, variant, seed):
    N = C.CONDITIONS[cond][1]
    if kind == "src":
        return refs.get(("N64", "Source", seed))
    return refs.get((f"N{N}", variant, seed))


# ---------------------------------------------------------------------------
# 2) Tüm koşumlar
# ---------------------------------------------------------------------------
def collect_rows(refs):
    groups = defaultdict(dict)
    for p in glob.glob(os.path.join(RAW_DIR, "*", "*", "*.npz")):
        if os.sep + "val" + os.sep in p:
            continue
        cond = os.path.basename(os.path.dirname(os.path.dirname(p)))
        variant = os.path.basename(os.path.dirname(p))
        stem = os.path.basename(p)[:-4]
        dom, s, st = stem.rsplit("_", 2)
        groups[(cond, variant, dom, int(st[2:]))][int(s[1:])] = p
    rows = []
    for (cond, variant, dom, st), seedmap in sorted(groups.items()):
        _, ydom = cached_domain(dom)
        data = {}
        for s, p in seedmap.items():
            logits, idx, _ = load_npz(p)
            order = np.argsort(idx)              # örnek kimliğine göre hizala
            data[s] = (logits[order], ydom[idx][order], idx)
        preds = {s: d[0].argmax(1) for s, d in data.items()}
        for s, (logits, y, idx) in data.items():
            r = {"cond": cond, "variant": variant, "domain": dom, "family": domain_family(dom),
                 "seed": s, "stream_seed": st, "shifted": dom != "clean"}
            r.update(all_metrics(logits, y))
            pr = logits.argmax(1)
            hist = np.bincount(pr, minlength=10) / len(pr)
            r["pred_marginal_entropy"] = float(-(hist[hist > 0] * np.log(hist[hist > 0])).sum())
            for c in range(10):
                m = y == c
                r[f"cls_acc_{c}"] = float((pr[m] == c).mean() * 100) if m.any() else np.nan
            others = [preds[o] for o in preds if o != s]
            for kind in REFS:
                ref = ref_for(refs, kind, cond, variant, s)
                if ref is None:
                    continue
                est = estimate_all(logits, ref, others if kind == "src" and others else None)
                for k, v in est.items():
                    if k == "GDE":
                        r["est_GDE"] = v
                    else:
                        r[f"est_{k}_{kind}"] = v
            rows.append(r)
    df = pd.DataFrame(rows)
    for col in [c for c in df.columns if c.startswith("est_")]:
        df["err" + col[3:]] = df[col] - df["acc"]
    return df


# ---------------------------------------------------------------------------
# yardımcılar
# ---------------------------------------------------------------------------
def main_view(df, cond, variant, st=C.MAIN_STREAM_SEED):
    return df[(df.cond == cond) & (df.variant == variant) & (df.stream_seed == st)]


def per_domain_seedmean(df, cond, variant, col, domains=SHIFTED):
    v = main_view(df, cond, variant)
    g = v.groupby("domain")[col].mean()
    return np.array([g[d] for d in domains])


def per_seed_domainmean(df, cond, variant, col, domains=SHIFTED):
    v = main_view(df, cond, variant)
    v = v[v.domain.isin(domains)]
    return v.groupby("seed")[col].mean().values


def domain_seed_matrix(df, cond, variant, col, domains=SHIFTED):
    v = main_view(df, cond, variant)
    p = v.pivot_table(index="domain", columns="seed", values=col)
    return p.loc[domains].values


def fmt(ms, nd=2):
    return f"{ms['mean']:.{nd}f} ± {ms['std']:.{nd}f}"


def write_table(name, df_table, note=""):
    df_table.to_csv(os.path.join(TABLES_DIR, f"{name}.csv"), index=False)
    with open(os.path.join(TABLES_DIR, f"{name}.md"), "w", encoding="utf-8") as f:
        if note:
            f.write(note + "\n\n")
        f.write(df_table.to_markdown(index=False) + "\n")


# ---------------------------------------------------------------------------
# Analizler
# ---------------------------------------------------------------------------
def source_models(val_acc):
    audit = json.load(open(os.path.join(LOGS_DIR, "audit_verification.json"), encoding="utf-8"))
    rows = []
    for s in C.MODEL_SEEDS:
        a = audit["V1_V2_checkpoints"].get(str(s), {})
        tr = os.path.join(LOGS_DIR, f"train_curves_seed{s}.json")
        tr_root = os.path.join(os.path.dirname(os.path.dirname(RESULTS_DIR)), "results", "logs",
                               f"cifar10_training_curves_seed{s}.json")
        src = tr if os.path.exists(tr) else (tr_root if os.path.exists(tr_root) else None)
        t_min = json.load(open(src))["training_time_minutes"] if src else None
        rows.append({"seed": s, "best_epoch": a.get("best_epoch_in_ckpt"),
                     "val_acc": a.get("val_acc_recomputed"), "clean_test_acc": a.get("clean_test_acc_recomputed"),
                     "train_minutes": None if t_min is None else round(t_min, 1),
                     "origin": ("orijinal depo (dondurulmuş)" + (" — süre DOĞRULANAMADI" if s == 42 else ""))
                     if s in (42, 43, 44) else "bu çalışmada eğitildi (GPU paylaşımlı; süre karşılaştırılamaz)"})
    t = pd.DataFrame(rows)
    write_table("T1_source_models", t)
    SUMMARY["source_models"] = rows
    SUMMARY["source_clean_acc"] = mean_std([r["clean_test_acc"] for r in rows if r["clean_test_acc"] is not None])


def rq1(df):
    out, tab = {}, []
    for cond in C.CONDITIONS:
        out[cond] = {}
        for m in METHODS:
            e = {}
            for col in ["acc", "ece", "nll", "brier", "overconf", "mean_conf"]:
                e[col] = mean_std(per_seed_domainmean(df, cond, m, col))
            e["clean_acc"] = mean_std(per_seed_domainmean(df, cond, m, "acc", ["clean"]))
            e["acc_hier_ci"] = hier_bootstrap_ci(domain_seed_matrix(df, cond, m, "acc"))
            out[cond][m] = e
            tab.append({"koşul": cond, "yöntem": m, "kaymalı doğruluk (%)": fmt(e["acc"]),
                        "temiz doğruluk (%)": fmt(e["clean_acc"]), "ECE (%)": fmt(e["ece"]),
                        "NLL": fmt(e["nll"], 3), "Brier": fmt(e["brier"], 3), "aşırı güven (pp)": fmt(e["overconf"])})
    write_table("T2_accuracy_calibration", pd.DataFrame(tab),
                "Kaymalı alanlar = 15 CIFAR-10-C (şiddet 5) + CIFAR-10.1. Değerler: 5 model tohumu üzerinden ortalama ± std "
                "(her tohumda 16 alanın ortalaması).")
    # alan bazlı tablo (iid64)
    rows = []
    for d in DOMAINS:
        r = {"alan": d, "aile": domain_family(d)}
        for m in METHODS:
            v = main_view(df, "iid64", m)
            r[m] = fmt(mean_std(v[v.domain == d]["acc"].values))
        rows.append(r)
    write_table("T3_per_domain_iid64", pd.DataFrame(rows), "iid64, doğruluk (%) ortalama ± std (5 model).")
    # H1
    a = per_domain_seedmean(df, "iid64", "TENT", "overconf")
    b = per_domain_seedmean(df, "iid64", "BN-adapt", "overconf")
    h1 = wilcoxon_paired(a, b)
    h1["decision"] = decide(h1["p"], h1["mean_diff"] > 0)
    out["H1"] = h1
    # ek: TENT-BN doğruluk farkı
    diff = domain_seed_matrix(df, "iid64", "TENT", "acc") - domain_seed_matrix(df, "iid64", "BN-adapt", "acc")
    out["tent_minus_bn_acc_iid64"] = {"mean": float(diff.mean()), "hier_ci": hier_bootstrap_ci(diff),
                                      **wilcoxon_paired(per_domain_seedmean(df, "iid64", "TENT", "acc"),
                                                        per_domain_seedmean(df, "iid64", "BN-adapt", "acc"))}
    SUMMARY["rq1"] = out


def decide(p_adj, direction_ok):
    if p_adj < 0.05 and direction_ok:
        return "desteklendi"
    if p_adj < 0.05 and not direction_ok:
        return "çürütüldü (ters yön)"
    return "desteklenmedi (kanıt yetersiz)"


def est_cols():
    # AC referans kullanmaz (src ve matched özdeştir) -> tek satır olarak raporlanır
    return [("AC", "src")] + [(e, r) for e in CONF_ESTIMATORS if e != "AC" for r in REFS] + [("GDE", None)]


def ecol(e, r):
    return "err_GDE" if e == "GDE" else f"err_{e}_{r}"


def rq2(df):
    df = df.copy()
    for c in [c for c in df.columns if c.startswith("err_")]:
        df["abs" + c[3:]] = df[c].abs()
    out = {}
    tab = []
    for cond in C.CONDITIONS:
        out[cond] = {}
        for m in METHODS:
            out[cond][m] = {}
            for e, r in est_cols():
                col = ecol(e, r)
                name = e if (r is None or e == "AC") else f"{e} ({r})"
                mae = mean_std(per_seed_domainmean(df, cond, m, "abs" + col[3:]))
                bias = mean_std(per_seed_domainmean(df, cond, m, col))
                est_name = "est" + col[3:]
                rho = spearman(per_domain_seedmean(df, cond, m, est_name), per_domain_seedmean(df, cond, m, "acc"))
                out[cond][m][name] = {"mae": mae, "bias": bias, "spearman": rho}
                tab.append({"koşul": cond, "yöntem": m, "tahmin edici": name, "MAE (pp)": fmt(mae),
                            "yanlılık (pp)": fmt(bias), "Spearman ρ": f"{rho:.3f}"})
    write_table("T4_estimator_mae", pd.DataFrame(tab),
                "MAE = |tahmin − gerçek doğruluk|, 16 kaymalı alan üzerinden; 5 model tohumu üzerinden ortalama ± std. "
                "src = kaynak modelin doğrulama kümesi referansı; matched = aynı TTA yöntemiyle işlenmiş doğrulama kümesi.")
    SUMMARY["rq2"] = out

    # H2
    pv, res = {}, {}
    for e in CONF_ESTIMATORS:
        a = per_domain_seedmean(df, "iid64", "TENT", f"abs_{e}_src")
        b = per_domain_seedmean(df, "iid64", "BN-adapt", f"abs_{e}_src")
        res[e] = wilcoxon_paired(a, b)
        pv[e] = res[e]["p"]
    adj = holm(pv)
    for e in res:
        res[e]["p_holm"] = adj[e]
        res[e]["decision"] = decide(adj[e], res[e]["mean_diff"] > 0)
    SUMMARY["H2"] = res

    # H3
    pv, res = {}, {}
    for m in ["BN-adapt", "TENT"]:
        cands = {f"{e} ({r})": per_domain_seedmean(df, "iid64", m, f"abs_{e}_{r}").mean()
                 for e in CONF_ESTIMATORS for r in REFS if not (e == "AC" and r == "matched")}
        best = min(cands, key=cands.get)
        e, r = best.split(" (")
        a = per_domain_seedmean(df, "iid64", m, "abs_GDE")
        b = per_domain_seedmean(df, "iid64", m, f"abs_{e}_{r[:-1]}")
        res[m] = {"best_conf_estimator": best, **wilcoxon_paired(a, b)}
        pv[m] = res[m]["p"]
    adj = holm(pv)
    for m in res:
        res[m]["p_holm"] = adj[m]
        res[m]["decision"] = decide(adj[m], res[m]["mean_diff"] < 0)
    SUMMARY["H3"] = res

    # H4
    pv, res = {}, {}
    for m in ["BN-adapt", "TENT"]:
        a = per_domain_seedmean(df, "sorted64", m, "acc")
        b = per_domain_seedmean(df, "sorted64", "Source", "acc")
        res[f"acc_{m}_vs_Source_sorted64"] = wilcoxon_paired(a, b)
        df_ = df.copy()
        df_["abs_conf_mean"] = df_[[f"abs_{e}_src" for e in CONF_ESTIMATORS]].mean(1)
        a = per_domain_seedmean(df_, "sorted64", m, "abs_conf_mean")
        b = per_domain_seedmean(df_, "iid64", m, "abs_conf_mean")
        res[f"confMAE_{m}_sorted64_vs_iid64"] = wilcoxon_paired(a, b)
    for k in res:
        pv[k] = res[k]["p"]
    adj = holm(pv)
    for k in res:
        res[k]["p_holm"] = adj[k]
        want = res[k]["mean_diff"] < 0 if k.startswith("acc_") else res[k]["mean_diff"] > 0
        res[k]["decision"] = decide(adj[k], want)
    SUMMARY["H4"] = res
    return df


def gates(df):
    """Etiketsiz uyarlama kapısı: (koşul, alan, model) başına tahmini en yüksek yöntemi seç."""
    v = df[(df.stream_seed == C.MAIN_STREAM_SEED) & (df.variant.isin(METHODS)) & (df.shifted)]
    est_names = ["GDE"] + [f"{e}_{r}" for e in CONF_ESTIMATORS for r in REFS if not (e == "AC" and r == "matched")]
    recs = []
    for (cond, dom, s), g in v.groupby(["cond", "domain", "seed"]):
        g = g.set_index("variant")
        if not all(m in g.index for m in METHODS):
            continue
        acc = g["acc"]
        best = acc.max()
        r = {"cond": cond, "domain": dom, "seed": s, "oracle": best}
        for m in METHODS:
            r[f"always_{m}"] = best - acc[m]
        for en in est_names:
            col = f"est_{en}"
            if col in g and g[col].notna().all():
                choice = g[col].idxmax()
                r[f"gate_{en}"] = best - acc[choice]
                r[f"choice_{en}"] = choice
        recs.append(r)
    G = pd.DataFrame(recs)
    pols = [f"always_{m}" for m in METHODS] + [f"gate_{en}" for en in est_names if f"gate_{en}" in G]
    tab, out = [], {}
    for p in pols:
        row = {"politika": p}
        for cond in C.CONDITIONS:
            gg = G[G.cond == cond]
            row[cond] = fmt(mean_std(gg.groupby("seed")[p].mean().values))
        allm = mean_std(G.groupby("seed")[p].mean().values)
        row["tüm koşullar"] = fmt(allm)
        out[p] = {"all": allm, **{c: mean_std(G[G.cond == c].groupby("seed")[p].mean().values) for c in C.CONDITIONS}}
        tab.append(row)
    write_table("T5_gate_regret", pd.DataFrame(tab),
                "Pişmanlık = oracle doğruluğu − seçilen yöntemin doğruluğu (pp); 16 kaymalı alan, 5 model; ortalama ± std (tohumlar).")
    unit = G.groupby(["cond", "domain"]).mean(numeric_only=True)
    fixed = {m: unit[f"always_{m}"].mean() for m in METHODS}
    best_fixed = min(fixed, key=fixed.get)
    h5 = wilcoxon_paired(unit["gate_GDE"].values, unit[f"always_{best_fixed}"].values)
    h5["best_fixed_policy"] = best_fixed
    h5["decision"] = decide(h5["p"], h5["mean_diff"] < 0)
    # GDE kapısının seçim dağılımı
    choices = G.groupby("cond")["choice_GDE"].value_counts(normalize=True).unstack(fill_value=0).round(3)
    SUMMARY["gates"] = {"policies": out, "H5": h5, "fixed_policy_mean_regret": fixed,
                        "gde_choice_freq": choices.to_dict(orient="index")}
    G.to_csv(os.path.join(TABLES_DIR, "gate_units.csv"), index=False)


def ablations(df):
    out = {}
    # (a) orijinal "güncelleme-sonrası tahmin" vs resmi protokol
    a = per_domain_seedmean(df, "iid64", "TENT-after", "acc", DOMAINS)
    b = per_domain_seedmean(df, "iid64", "TENT", "acc", DOMAINS)
    out["after_vs_official_acc"] = wilcoxon_paired(a, b)
    out["after_vs_official_acc"]["per_domain_mean_diff"] = dict(zip(DOMAINS, np.round(a - b, 3).tolist()))
    a = per_domain_seedmean(df, "iid64", "TENT-after", "ece", DOMAINS)
    b = per_domain_seedmean(df, "iid64", "TENT", "ece", DOMAINS)
    out["after_vs_official_ece"] = wilcoxon_paired(a, b)
    # (b) öğrenme oranı
    tab = []
    for v in ["TENT", "TENT-lr1e-2", "TENT-after", "BN-adapt"]:
        e = {k: mean_std(per_seed_domainmean(df, "iid64", v, k)) for k in ["acc", "ece", "overconf"]}
        dd = df.copy()
        for col in ["err_ATC-MC_src", "err_ATC-MC_matched", "err_GDE", "err_DoC_src"]:
            dd["a" + col] = dd[col].abs()
        for col in ["aerr_ATC-MC_src", "aerr_ATC-MC_matched", "aerr_GDE", "aerr_DoC_src"]:
            e[col] = mean_std(per_seed_domainmean(dd, "iid64", v, col))
        out[v] = e
        tab.append({"varyant": v, "doğruluk (%)": fmt(e["acc"]), "ECE (%)": fmt(e["ece"]),
                    "aşırı güven (pp)": fmt(e["overconf"]), "MAE ATC-MC src": fmt(e["aerr_ATC-MC_src"]),
                    "MAE ATC-MC matched": fmt(e["aerr_ATC-MC_matched"]), "MAE DoC src": fmt(e["aerr_DoC_src"]),
                    "MAE GDE": fmt(e["aerr_GDE"])})
    write_table("T6_ablation_tent", pd.DataFrame(tab), "iid64, 16 kaymalı alan, 5 model; ortalama ± std.")
    # (c) referans türü: src vs matched
    pv, res = {}, {}
    for m in ["BN-adapt", "TENT"]:
        for e in [x for x in CONF_ESTIMATORS if x != "AC"]:
            dd = df.copy()
            a = per_domain_seedmean(dd.assign(x=dd[f"err_{e}_matched"].abs()), "iid64", m, "x")
            b = per_domain_seedmean(dd.assign(x=dd[f"err_{e}_src"].abs()), "iid64", m, "x")
            k = f"{m}|{e}"
            res[k] = wilcoxon_paired(a, b)
            pv[k] = res[k]["p"]
    adj = holm(pv)
    for k in res:
        res[k]["p_holm"] = adj[k]
    out["ref_matched_vs_src"] = res
    # (d) akış sırası varyansı vs model varyansı (model 42 / iid64)
    sv = []
    for m in C.STREAM_METHODS:
        st = df[(df.cond == "iid64") & (df.variant == m) & (df.seed == C.STREAM_ABLATION_MODEL) & df.shifted
                & (df.stream_seed.isin(C.STREAM_ABLATION_SEEDS + [C.MAIN_STREAM_SEED]))]
        stream_sd = st.groupby("domain")["acc"].std(ddof=1).mean()
        mv = main_view(df, "iid64", m)
        model_sd = mv[mv.shifted].groupby("domain")["acc"].std(ddof=1).mean()
        sv.append({"yöntem": m, "akış sırası std (pp, 4 akış, model 42)": round(float(stream_sd), 3),
                   "model tohumu std (pp, 5 model)": round(float(model_sd), 3)})
    write_table("T7_stream_vs_model_variance", pd.DataFrame(sv), "Alan başına std'nin 16 kaymalı alan üzerinden ortalaması.")
    out["stream_vs_model_sd"] = sv
    SUMMARY["ablations"] = out


def error_analysis(df):
    out = {}
    rows = []
    for cond in ["iid64", "sorted64"]:
        for m in METHODS:
            v = main_view(df, cond, m)
            v = v[v.shifted]
            for c in range(10):
                rows.append({"koşul": cond, "yöntem": m, "sınıf": CIFAR10_CLASSES[c],
                             "doğruluk": round(float(v[f"cls_acc_{c}"].mean()), 2)})
    t = pd.DataFrame(rows).pivot_table(index="sınıf", columns=["koşul", "yöntem"], values="doğruluk")
    t = t.reindex(list(CIFAR10_CLASSES))
    t.columns = [f"{a}/{b}" for a, b in t.columns]
    t = t.reset_index()
    write_table("T8_per_class_accuracy", t, "Sınıf bazlı doğruluk (%), 16 kaymalı alan ve 5 model ortalaması.")
    out["per_class"] = t.to_dict(orient="records")
    # tahmin marjinali entropisi (çöküş göstergesi; maksimum ln10 = 2.303)
    out["pred_marginal_entropy"] = {cond: {m: mean_std(per_seed_domainmean(df, cond, m, "pred_marginal_entropy"))
                                           for m in METHODS} for cond in C.CONDITIONS}
    # en büyük tahmin hataları (iid64 + sorted64, birincil tahmin ediciler)
    v = df[(df.stream_seed == C.MAIN_STREAM_SEED) & df.variant.isin(METHODS) & df.shifted]
    agg = v.groupby(["cond", "variant", "domain"])[["acc", "est_GDE", "est_ATC-MC_src", "est_DoC_src"]].mean().reset_index()
    for e in ["GDE", "ATC-MC_src", "DoC_src"]:
        agg[f"err_{e}"] = agg[f"est_{e}"] - agg["acc"]
    worst = {}
    for e in ["GDE", "ATC-MC_src", "DoC_src"]:
        w = agg.reindex(agg[f"err_{e}"].abs().sort_values(ascending=False).index).head(6)
        worst[e] = w[["cond", "variant", "domain", "acc", f"est_{e}", f"err_{e}"]].round(2).to_dict(orient="records")
    out["worst_errors"] = worst
    # GDE hata vs gerçek doğruluk ilişkisi (yüksek doğrulukta uyuşma ↔ doğruluk)
    out["gde_err_vs_acc_spearman"] = spearman(agg["acc"], agg["err_GDE"])
    agg.round(3).to_csv(os.path.join(TABLES_DIR, "domain_level_estimates.csv"), index=False)
    SUMMARY["error_analysis"] = out


def main():
    refs, val_acc = build_refs()
    SUMMARY["val_acc"] = {f"{k[0]}|{k[1]}|{k[2]}": round(v, 2) for k, v in val_acc.items()}
    df = collect_rows(refs)
    df.to_csv(os.path.join(RESULTS_DIR, "all_runs.csv"), index=False)
    SUMMARY["n_runs"] = int(len(df))
    SUMMARY["n_runs_by_cond_variant"] = {f"{c}|{v}": int(n) for (c, v), n in df.groupby(["cond", "variant"]).size().items()}
    SUMMARY["seeds_present"] = sorted(int(s) for s in df.seed.unique())
    source_models(val_acc)
    rq1(df)
    rq2(df)
    gates(df)
    ablations(df)
    error_analysis(df)
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(SUMMARY, f, indent=2, ensure_ascii=False, default=float)
    print(json.dumps({k: SUMMARY[k] for k in ["n_runs", "seeds_present"]}, indent=1))
    for k in ["H2", "H3", "H4"]:
        print(k, json.dumps(SUMMARY[k], indent=1, ensure_ascii=False, default=float)[:1500])
    print("H1", SUMMARY["rq1"]["H1"])
    print("H5", SUMMARY["gates"]["H5"])


if __name__ == "__main__":
    main()
