"""NetShield final evaluation. Usage: run_final.py dev|final [cache_dir]

Regimes
  baseline  earlier project method (RF, 11 payload features, source-derived threshold)
  transfer  no target labels at all
  joint     one model trained with labels from both datasets
  few       transfer plus a small labelled sample of the target
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy.stats import rankdata
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import features as F

warnings.filterwarnings("ignore")
SPLIT = sys.argv[1]
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / "cache"
OUT = ROOT / "outputs" / "final_model" / ("tables" if SPLIT == "final" else "tables_dev")
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
DEV_DAYS = ["Monday-WorkingHours", "Tuesday-WorkingHours", "Wednesday-workingHours"]
TRANSFER_COLS = F.columns(["size", "rate", "shape"])
JOINT_COLS = F.columns(["size", "rate", "shape", "proto", "service", "timing"])
CONT = [c for c in TRANSFER_COLS if c not in F.NOT_SCALED]
OLD_11 = ["flow_duration_sec", "fwd_packets", "bwd_packets", "fwd_bytes", "bwd_bytes", "total_packets",
          "total_bytes", "packet_rate", "fwd_packet_rate", "bwd_packet_rate", "avg_packet_size"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def metrics(y, score, thr, **tags):
    h = score >= thr
    tn = int(((~h) & (y == 0)).sum()); fp = int((h & (y == 0)).sum()); fn = int(((~h) & (y == 1)).sum()); tp = int((h & (y == 1)).sum())
    return {**tags, "n": len(y), "attack_rate": float(y.mean()), "roc_auc": roc_auc_score(y, score),
            "pr_auc": average_precision_score(y, score), "f1": f1_score(y, h, zero_division=0),
            "precision": precision_score(y, h, zero_division=0), "recall": recall_score(y, h, zero_division=0),
            "fpr": fp / max(fp + tn, 1), "flagged_rate": float(h.mean()), "threshold": float(thr), "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def trivial_f1(y):
    return 2 * y.mean() / (1 + y.mean())


def lgbm(seed, leaves=63, trees=200, weighted=True):
    return LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=leaves, min_child_samples=100, subsample=0.8,
                          subsample_freq=1, colsample_bytree=0.8, n_jobs=8, random_state=seed, verbose=-1,
                          class_weight="balanced" if weighted else None)


def best_f1_threshold(y, p):
    ts = np.linspace(0.02, 0.98, 97)
    return float(ts[int(np.argmax([f1_score(y, p >= t, zero_division=0) for t in ts]))])


def otsu(p):
    h, e = np.histogram(p, bins=200)
    h = h / h.sum(); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
    return float(c[int(np.argmax((mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)))])


def robust_norm(X, pool):
    med, iqr = pool[CONT].median(), (pool[CONT].quantile(0.75) - pool[CONT].quantile(0.25)).replace(0, 1)
    X = X.copy(); X[CONT] = (X[CONT] - med) / iqr
    return X


def quantile_norm(X, pool):
    X = X.copy()
    for k in CONT:
        ref = np.sort(pool[k].dropna().values[:: max(1, len(pool) // 200000)])
        X[k] = (np.searchsorted(ref, X[k].values, side="right") / len(ref)).astype("float32")
    return X


NORMS = {"robust": robust_norm, "quantile": quantile_norm}


def fit_ensemble(Xs, ys, pool_s, pool_t, Xt_list, seed=SEED, extra=None):
    """Six models (2 normalizations x 3 sizes), each normalized per dataset. Returns rank-averaged target scores."""
    sums = [np.zeros(len(X)) for X in Xt_list]
    idx = np.random.default_rng(seed).choice(len(Xs), min(500_000, len(Xs)), replace=False)
    for nname, nf in NORMS.items():
        Xsn = nf(Xs.iloc[idx], pool_s)[TRANSFER_COLS]; ysn = ys[idx]
        Xtn = [nf(X, pool_t)[TRANSFER_COLS] for X in Xt_list]
        w = None
        if extra is not None:
            Xe = nf(extra[0], pool_t)[TRANSFER_COLS]
            Xsn = pd.concat([Xsn, Xe]); ysn = np.r_[ysn, extra[1]]
            w = np.r_[np.ones(len(idx)), np.full(len(Xe), len(idx) / max(len(Xe), 1))]
        for leaves in [7, 15, 63]:
            m = lgbm(seed, leaves=leaves, trees=150 if extra is not None else 200).fit(Xsn, ysn, sample_weight=w)
            for i, X in enumerate(Xtn):
                sums[i] += rankdata(m.predict_proba(X)[:, 1]) / len(X)
    return [s / 6 for s in sums]


# ---------------------------------------------------------------- data
log("loading", SPLIT)
cic = pd.read_parquet(CACHE / "cicids.parquet")
cic = cic.drop_duplicates([c for c in cic.columns if c not in ("capture", "label_name")]).reset_index(drop=True)
unsw = pd.concat([pd.read_parquet(CACHE / "unsw_train.parquet").assign(part="train"),
                  pd.read_parquet(CACHE / "unsw_test.parquet").assign(part="test")], ignore_index=True)
raw = pd.read_parquet(CACHE / "unsw_raw.parquet")
raw = raw.drop_duplicates([c for c in raw.columns if c != "part"]).reset_index(drop=True)
Xc, yc = F.from_cicids(cic, True), cic["label"].values
Xc_pay = F.from_cicids(cic, False)
Xu, yu = F.from_unsw(unsw), unsw["label"].values
Xr, yr = F.from_unsw(raw), raw["label"].values
c_mask = cic["capture"].isin(DEV_DAYS).values if SPLIT == "dev" else ~cic["capture"].isin(DEV_DAYS).values
u_mask = (unsw["part"] == "train").values if SPLIT == "dev" else (unsw["part"] == "test").values
r_mask = raw["part"].isin([1, 2]).values if SPLIT == "dev" else raw["part"].isin([3, 4]).values
log("frozen sets: cicids", int(c_mask.sum()), yc[c_mask].mean(), "unsw", int(u_mask.sum()), yu[u_mask].mean(), "unsw raw", int(r_mask.sum()), yr[r_mask].mean())
results, families = [], []


def old_frame(X, payload_source):
    """Rebuild the earlier 11-feature view from the log features (bytes use the earlier payload definition for CICIDS)."""
    d = pd.DataFrame(index=X.index)
    d["flow_duration_sec"] = np.expm1(X["log_dur"]) / 1e6
    d["fwd_packets"], d["bwd_packets"] = np.expm1(X["log_fwd_pkts"]), np.expm1(X["log_bwd_pkts"])
    d["fwd_bytes"], d["bwd_bytes"] = np.expm1(payload_source["log_fwd_bytes"]), np.expm1(payload_source["log_bwd_bytes"])
    d["total_packets"], d["total_bytes"] = d.fwd_packets + d.bwd_packets, d.fwd_bytes + d.bwd_bytes
    safe = d.flow_duration_sec.replace(0, np.nan)
    d["packet_rate"], d["fwd_packet_rate"], d["bwd_packet_rate"] = d.total_packets / safe, d.fwd_packets / safe, d.bwd_packets / safe
    d["avg_packet_size"] = d.total_bytes / d.total_packets.replace(0, np.nan)
    return d[OLD_11].replace([np.inf, -np.inf], np.nan).astype("float64")


def coral_fit(Xs, Xt):
    def p(C, power):
        v, Q = np.linalg.eigh(C); v = np.clip(v, 1e-6, None)
        return (Q * v ** power) @ Q.T
    return Xs.mean(0), Xt.mean(0), p(np.cov(Xs, rowvar=False), -0.5) @ p(np.cov(Xt, rowvar=False), 0.5)


def run_baseline(direction, Xs, ys, Xt, yt, Xt_pool):
    idx = np.random.default_rng(SEED).choice(len(Xs), min(300_000, len(Xs)), replace=False)
    a, b = train_test_split(idx, test_size=0.2, stratify=ys[idx], random_state=SEED)
    imp = SimpleImputer(strategy="median").fit(Xs.iloc[a])
    rf = dict(n_estimators=200, min_samples_leaf=2, max_features="sqrt", n_jobs=8, random_state=SEED)
    m = RandomForestClassifier(**rf).fit(imp.transform(Xs.iloc[a]), ys[a])
    t = best_f1_threshold(ys[b], m.predict_proba(imp.transform(Xs.iloc[b]))[:, 1])
    results.append(metrics(yt, m.predict_proba(imp.transform(Xt))[:, 1], t, regime="baseline", direction=direction, method="RF 11 payload features, source threshold"))
    sc = StandardScaler().fit(imp.transform(Xs.iloc[a]))
    Zs, Zv, Zt, Zp = (sc.transform(imp.transform(z)) for z in (Xs.iloc[a], Xs.iloc[b], Xt, Xt_pool))
    ms, mt, A = coral_fit(Zs, Zp[:: max(1, len(Zp) // 300000)])
    m = RandomForestClassifier(**rf, class_weight="balanced").fit((Zs - ms) @ A + mt, ys[a])
    t = best_f1_threshold(ys[b], m.predict_proba((Zv - ms) @ A + mt)[:, 1])
    results.append(metrics(yt, m.predict_proba(Zt)[:, 1], t, regime="baseline", direction=direction, method="RF + CORAL + balanced, source threshold"))
    log("baseline", direction, [(r["method"][:12], round(r["roc_auc"], 3), round(r["f1"], 3)) for r in results[-2:]])


old_u, old_c = old_frame(Xu, Xu), old_frame(Xc, Xc_pay)
run_baseline("UNSW->CICIDS", old_u, yu, old_c[c_mask], yc[c_mask], old_c)
run_baseline("CICIDS->UNSW", old_c, yc, old_u[u_mask], yu[u_mask], old_frame(Xr, Xr))

# ---------------------------------------------------------------- strict transfer: no target labels anywhere
for direction, Xs, ys, ps, Xt_sets, pt in [
    ("UNSW->CICIDS", Xr, yr, Xr, [("all frozen days", Xc[c_mask], yc[c_mask], cic.loc[c_mask, "label_name"].values)], Xc),
    ("CICIDS->UNSW", Xc, yc, Xc, [("official file", Xu[u_mask], yu[u_mask], unsw.loc[u_mask, "attack_cat"].astype(str).str.strip().values),
                                  ("full capture", Xr[r_mask], yr[r_mask], raw.loc[r_mask, "attack_cat"].values)], Xr),
]:
    scores = fit_ensemble(Xs, ys, ps, pt, [x[1] for x in Xt_sets])
    for (name, Xt, yt, fam), s in zip(Xt_sets, scores):
        t = otsu(s)
        results.append(metrics(yt, s, t, regime="transfer", direction=direction, method=f"ensemble, normalized, unlabeled threshold ({name})", trivial_f1=trivial_f1(yt)))
        log(direction, name, round(results[-1]["roc_auc"], 3), round(results[-1]["f1"], 3), round(results[-1]["recall"], 3), round(results[-1]["fpr"], 3))
        benign = yt == 0
        for f in sorted(set(fam[yt == 1])):
            sel = fam == f
            families.append({"method": "ensemble", "direction": direction, "eval": name, "family": f, "n": int(sel.sum()), "recall": float((s[sel] >= t).mean()),
                             "roc_vs_benign": roc_auc_score(np.r_[np.ones(sel.sum()), np.zeros(benign.sum())], np.r_[s[sel], s[benign]])})


def run_coral_log(direction, Xs, ys, Xpool, Xt_sets, label, regime):
    """Selected method: CORAL + balanced RF on corrected log features, aligned to all unlabeled target traffic."""
    idx = np.random.default_rng(SEED).choice(len(Xs), min(300_000, len(Xs)), replace=False)
    a, b = train_test_split(idx, test_size=0.2, stratify=ys[idx], random_state=SEED)
    imp = SimpleImputer(strategy="median").fit(Xs.iloc[a][TRANSFER_COLS])
    sc = StandardScaler().fit(imp.transform(Xs.iloc[a][TRANSFER_COLS]))
    z = lambda X: sc.transform(imp.transform(X[TRANSFER_COLS]))
    Zs, Zv, Zp = z(Xs.iloc[a]), z(Xs.iloc[b]), z(Xpool)[:: max(1, len(Xpool) // 300000)]
    ms, mt, A = coral_fit(Zs, Zp)
    m = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, max_features="sqrt", n_jobs=8, random_state=SEED, class_weight="balanced").fit((Zs - ms) @ A + mt, ys[a])
    source_t = best_f1_threshold(ys[b], m.predict_proba((Zv - ms) @ A + mt)[:, 1])
    for name, Xt, yt, fam in Xt_sets:
        s = m.predict_proba(z(Xt))[:, 1]
        t = otsu(s)
        results.append(metrics(yt, s, t, regime=regime, direction=direction, method=f"{label}, unlabeled threshold ({name})", trivial_f1=trivial_f1(yt), source_threshold_f1=f1_score(yt, s >= source_t, zero_division=0)))
        log(label, direction, name, round(results[-1]["roc_auc"], 3), round(results[-1]["pr_auc"], 3), round(results[-1]["f1"], 3), round(results[-1]["recall"], 3), round(results[-1]["fpr"], 3))
        if regime == "transfer_selected":
            benign = yt == 0
            for f in sorted(set(fam[yt == 1])):
                sel = fam == f
                families.append({"method": "selected", "direction": direction, "eval": name, "family": f, "n": int(sel.sum()), "recall": float((s[sel] >= t).mean()),
                                 "roc_vs_benign": roc_auc_score(np.r_[np.ones(sel.sum()), np.zeros(benign.sum())], np.r_[s[sel], s[benign]])})


cic_eval = [("all frozen days", Xc[c_mask], yc[c_mask], cic.loc[c_mask, "label_name"].values)]
unsw_eval = [("official file", Xu[u_mask], yu[u_mask], unsw.loc[u_mask, "attack_cat"].astype(str).str.strip().values),
             ("full capture", Xr[r_mask], yr[r_mask], raw.loc[r_mask, "attack_cat"].values)]
run_coral_log("UNSW->CICIDS", Xu, yu, Xc, cic_eval, "CORAL RF corrected log features", "transfer_selected")
run_coral_log("CICIDS->UNSW", Xc, yc, Xr, unsw_eval, "CORAL RF corrected log features", "transfer_selected")
run_coral_log("UNSW->CICIDS", Xr, yr, Xc, cic_eval, "same, full UNSW capture as source (not selected)", "transfer_alternative")
pd.DataFrame(families).to_csv(OUT / "transfer_by_attack_family.csv", index=False)
pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

# ---------------------------------------------------------------- joint: labels from both datasets, one model, one threshold
log("joint")
key_c = pd.util.hash_pandas_object(Xc[JOINT_COLS].round(4), index=False).values
_, first = np.unique(key_c, return_index=True)
ci = np.sort(first)
ctr, cte = train_test_split(ci, test_size=0.3, stratify=yc[ci], random_state=SEED)
ctr, cva = train_test_split(ctr, test_size=0.15, stratify=yc[ctr], random_state=SEED)
utrain = np.where((unsw["part"] == "train").values)[0]
utest = np.where((unsw["part"] == "test").values)[0]
key_u = pd.util.hash_pandas_object(Xu[JOINT_COLS].round(4), index=False).values
utest = utest[~np.isin(key_u[utest], key_u[utrain])]
_, u_first = np.unique(key_u[utest], return_index=True); utest = utest[np.sort(u_first)]
_, u_first = np.unique(key_u[utrain], return_index=True); utrain = utrain[np.sort(u_first)]
utr, uva = train_test_split(utrain, test_size=0.15, stratify=yu[utrain], random_state=SEED)
log("joint sizes after removing overlap: cic train", len(ctr), "test", len(cte), "unsw train", len(utr), "test", len(utest))
cols = JOINT_COLS
if SPLIT == "dev":
    cte, utest = cte[: len(cte) // 2], utest[: len(utest) // 2]
else:
    cte, utest = cte[len(cte) // 2 :], utest[len(utest) // 2 :]


def weights(n_u, n_c):
    return np.r_[np.full(n_u, 0.5 / n_u), np.full(n_c, 0.5 / n_c)] * (n_u + n_c)


ctr_fit = np.random.default_rng(SEED).choice(ctr, min(700_000, len(ctr)), replace=False)
Xj = pd.concat([Xu.iloc[utr][cols], Xc.iloc[ctr_fit][cols]]); yj = np.r_[yu[utr], yc[ctr_fit]]
m = lgbm(SEED, leaves=127, trees=400).fit(Xj, yj, sample_weight=weights(len(utr), len(ctr_fit)))
pv = np.r_[m.predict_proba(Xu.iloc[uva][cols])[:, 1], m.predict_proba(Xc.iloc[cva][cols])[:, 1]]
yv = np.r_[yu[uva], yc[cva]]
wv = np.r_[np.full(len(uva), 0.5 / len(uva)), np.full(len(cva), 0.5 / len(cva))]
ts = np.linspace(0.02, 0.98, 97)
t_joint = float(ts[int(np.argmax([f1_score(yv, pv >= t, sample_weight=wv) for t in ts]))])
p_u, p_c = m.predict_proba(Xu.iloc[utest][cols])[:, 1], m.predict_proba(Xc.iloc[cte][cols])[:, 1]
results.append(metrics(yu[utest], p_u, t_joint, regime="joint", direction="both", method="joint model, tested on UNSW"))
results.append(metrics(yc[cte], p_c, t_joint, regime="joint", direction="both", method="joint model, tested on CICIDS"))
log("joint", round(results[-2]["f1"], 3), round(results[-1]["f1"], 3), "t", t_joint)

for name, tr, va, te, Xd, yd in [("UNSW", utr, uva, utest, Xu, yu), ("CICIDS", ctr_fit, cva, cte, Xc, yc)]:
    mm = lgbm(SEED, leaves=127, trees=400).fit(Xd.iloc[tr][cols], yd[tr])
    t = best_f1_threshold(yd[va], mm.predict_proba(Xd.iloc[va][cols])[:, 1])
    results.append(metrics(yd[te], mm.predict_proba(Xd.iloc[te][cols])[:, 1], t, regime="within", direction=name, method=f"separate model, trained and tested on {name}"))
    log("within", name, round(results[-1]["f1"], 3))

# Harder CICIDS test: joint model never sees the labelled attack days it is scored on.
hold = c_mask
ctr_day = np.where(~hold)[0]
ctr_day = np.random.default_rng(SEED).choice(ctr_day, min(700_000, len(ctr_day)), replace=False)
Xj = pd.concat([Xu.iloc[utr][cols], Xc.iloc[ctr_day][cols]]); yj = np.r_[yu[utr], yc[ctr_day]]
md = lgbm(SEED, leaves=127, trees=400).fit(Xj, yj, sample_weight=weights(len(utr), len(ctr_day)))
results.append(metrics(yc[hold], md.predict_proba(Xc.loc[hold, cols])[:, 1], t_joint, regime="joint_unseen_days", direction="both", method="joint model, CICIDS test days excluded from training"))
log("joint unseen days", round(results[-1]["f1"], 3), round(results[-1]["roc_auc"], 3))
pd.DataFrame(results).to_csv(OUT / "results.csv", index=False)

# ---------------------------------------------------------------- few labelled target rows
log("few labels")
few = []
for direction, Xs, ys, ps, pt, Xpool, ypool in [
    ("UNSW->CICIDS", Xr, yr, Xr, Xc, Xc[c_mask], yc[c_mask]),
    ("CICIDS->UNSW", Xc, yc, Xc, Xr, Xu[u_mask], yu[u_mask]),
]:
    for n in [100, 500, 2000]:
        for seed in [1, 2, 3]:
            rs = np.random.default_rng(seed)
            lab = rs.choice(len(Xpool), n, replace=False)
            rest = np.setdiff1d(np.arange(len(Xpool)), lab)
            Xlab, ylab = Xpool.iloc[lab], ypool[lab]
            oof = np.zeros(n)
            for tr, va in StratifiedKFold(3, shuffle=True, random_state=seed).split(lab, ylab):
                oof[va] = fit_ensemble(Xs, ys, ps, pt, [Xlab.iloc[va]], seed=seed, extra=(Xlab.iloc[tr], ylab[tr]))[0]
            s = fit_ensemble(Xs, ys, ps, pt, [Xpool.iloc[rest]], seed=seed, extra=(Xlab, ylab))[0]
            few.append(metrics(ypool[rest], s, best_f1_threshold(ylab, oof), regime="few_labels", direction=direction, method=f"{n} labelled target flows", labelled=n, seed=seed, trivial_f1=trivial_f1(ypool[rest])))
            log(direction, n, seed, round(few[-1]["roc_auc"], 3), round(few[-1]["f1"], 3), round(few[-1]["fpr"], 3))
pd.DataFrame(few).to_csv(OUT / "few_labels.csv", index=False)
json.dump({"split": SPLIT, "cicids_days_in_this_split": sorted(set(cic.loc[c_mask, "capture"])), "seed": SEED,
           "transfer_features": TRANSFER_COLS, "joint_features": JOINT_COLS}, open(OUT / "config.json", "w"), indent=1)
log("done")
