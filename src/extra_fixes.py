"""Extra transfer experiments. Usage: extra_fixes.py grid|final [cache_dir]

grid   dev data only. Scores every configuration and locks one per scope in selected.json.
       Rule: highest mean dev ROC-AUC over both directions.
final  scores only the locked configurations on the final data.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent))
import features as F

warnings.filterwarnings("ignore")
MODE = sys.argv[1]
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / "cache"
OUT = ROOT / "outputs" / "final_model" / "extra"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 42
DEV_DAYS = ["Monday-WorkingHours", "Tuesday-WorkingHours", "Wednesday-workingHours"]
CIC_SHARED = {"DoS Hulk", "DoS GoldenEye", "DoS slowloris", "DoS Slowhttptest", "DDoS", "PortScan"}
UNSW_SHARED = {"DoS", "Reconnaissance"}
RATIO = F.columns(["rate", "shape"])
VIEWS = {"counts+rates+ratios": F.columns(["size", "rate", "shape"]), "rates+ratios": RATIO, "rates+ratios+duration": ["log_dur"] + RATIO}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


cic = pd.read_parquet(CACHE / "cicids.parquet")
cic = cic.drop_duplicates([c for c in cic.columns if c not in ("capture", "label_name")]).reset_index(drop=True)
unsw = pd.concat([pd.read_parquet(CACHE / "unsw_train.parquet").assign(part="train"), pd.read_parquet(CACHE / "unsw_test.parquet").assign(part="test")], ignore_index=True)
raw = pd.read_parquet(CACHE / "unsw_raw.parquet")
raw = raw.drop_duplicates([c for c in raw.columns if c != "part"]).reset_index(drop=True)
Xc, yc, Xu, yu, Xr, yr = F.from_cicids(cic, True), cic["label"].values, F.from_unsw(unsw), unsw["label"].values, F.from_unsw(raw), raw["label"].values
cic_s = (yc == 0) | cic["label_name"].isin(CIC_SHARED).values
uns_s = (yu == 0) | unsw["attack_cat"].astype(str).str.strip().isin(UNSW_SHARED).values
raw_s = (yr == 0) | raw["attack_cat"].astype(str).str.strip().isin(UNSW_SHARED).values
SHARED = {id(Xc): cic_s, id(Xu): uns_s, id(Xr): raw_s}
c_dev = cic["capture"].isin(DEV_DAYS).values
u_dev = (unsw["part"] == "train").values


def cont(cols):
    return [c for c in cols if c not in F.NOT_SCALED]


def quantile(X, pool, cols):
    X = X[cols].copy()
    for k in cont(cols):
        ref = np.sort(pool[k].dropna().values[:: max(1, len(pool) // 200000)])
        X[k] = (np.searchsorted(ref, X[k].values, side="right") / len(ref)).astype("float32")
    return X


def coral_map(Zs, Zp):
    def p(C, power):
        v, Q = np.linalg.eigh(C); v = np.clip(v, 1e-6, None)
        return (Q * v ** power) @ Q.T
    ms, mt, A = Zs.mean(0), Zp.mean(0), p(np.cov(Zs, rowvar=False), -0.5) @ p(np.cov(Zp, rowvar=False), 0.5)
    return lambda Z: (Z - ms) @ A + mt


def fit_predict(align, Xs, ys, pool_s, pool_t, Xt_list, cols, n=300_000, trees=200, seed=SEED):
    idx = np.random.default_rng(seed).choice(len(Xs), min(n, len(Xs)), replace=False)
    if align == "quantile":
        m = LGBMClassifier(n_estimators=trees, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=8, random_state=seed, verbose=-1, class_weight="balanced")
        m.fit(quantile(Xs.iloc[idx], pool_s, cols), ys[idx])
        return [m.predict_proba(quantile(X, pool_t, cols))[:, 1] for X in Xt_list]
    imp = SimpleImputer(strategy="median").fit(Xs.iloc[idx][cols])
    sc = StandardScaler().fit(imp.transform(Xs.iloc[idx][cols]))
    z = lambda X: sc.transform(imp.transform(X[cols]))
    Zs = z(Xs.iloc[idx])
    g = coral_map(Zs, z(pool_t)[:: max(1, len(pool_t) // 300000)])
    m = RandomForestClassifier(n_estimators=trees, min_samples_leaf=2, max_features="sqrt", n_jobs=8, random_state=seed, class_weight="balanced").fit(g(Zs), ys[idx])
    return [m.predict_proba(z(X))[:, 1] for X in Xt_list]


def prune_by_domain(Xs, Xt, cols, stop=0.75, min_features=6):
    """Adversarial validation: remove the feature that most separates the datasets until they are harder to tell apart."""
    rng = np.random.default_rng(SEED); kept = list(cols); trail = []
    a, b = rng.choice(len(Xs), min(100_000, len(Xs)), replace=False), rng.choice(len(Xt), min(100_000, len(Xt)), replace=False)
    Qs, Qt = quantile(Xs.iloc[a], Xs, cols), quantile(Xt.iloc[b], Xt, cols)
    D = pd.concat([Qs, Qt]); y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    tr, te = train_test_split(np.arange(len(y)), test_size=0.3, stratify=y, random_state=SEED)
    while True:
        m = LGBMClassifier(n_estimators=100, num_leaves=31, n_jobs=8, random_state=SEED, verbose=-1).fit(D.iloc[tr][kept], y[tr])
        auc = roc_auc_score(y[te], m.predict_proba(D.iloc[te][kept])[:, 1])
        top = kept[int(np.argmax(m.booster_.feature_importance("gain")))]
        trail.append({"n_features": len(kept), "domain_auc": auc, "most_separating_feature": top})
        if auc < stop or len(kept) <= min_features:
            return kept, trail
        kept.remove(top)


def subset(X, y, scope):
    if scope == "all":
        return X, y
    mask = SHARED[id(X)]
    return X[mask], y[mask]


def transfer_rocs(cfg, Xud_list=None):
    """Dev ROC-AUC for both directions under one configuration."""
    align, cols, scope = cfg
    out = {}
    for name, Xs, ys in [("official", Xu, yu), ("full", Xr, yr)]:
        S, Sy = subset(Xs, ys, scope)
        mask = c_dev if scope == "all" else c_dev & cic_s
        p = fit_predict(align, S, Sy, Xs, Xc, [Xc[mask]], cols)[0]
        out[name] = (roc_auc_score(yc[mask], p), average_precision_score(yc[mask], p))
    S, Sy = subset(Xc, yc, scope)
    mask = u_dev if scope == "all" else u_dev & uns_s
    p = fit_predict(align, S, Sy, Xc, Xr, [Xu[mask]], cols)[0]
    out["c2u"] = (roc_auc_score(yu[mask], p), average_precision_score(yu[mask], p))
    return out


if MODE == "grid":
    views = dict(VIEWS)
    trails = []
    raw_auc = {}
    for pair, Xs in [("official UNSW vs CICIDS", Xu), ("full UNSW vs CICIDS", Xr)]:
        kept, tr = prune_by_domain(Xs, Xc, VIEWS["counts+rates+ratios"])
        views[f"domain-pruned ({pair.split()[0]})"] = kept
        trails += [{"pair": pair, **t} for t in tr]
        log(pair, "domain AUC", round(tr[0]["domain_auc"], 3), "->", round(tr[-1]["domain_auc"], 3), "kept", len(kept))
    pd.DataFrame(trails).to_csv(OUT / "adversarial_validation.csv", index=False)
    rows = []
    for scope in ["all", "shared"]:
        for align in ["quantile", "coral"]:
            for vname, cols in views.items():
                r = transfer_rocs((align, cols, scope))
                src = "official" if r["official"][0] >= r["full"][0] else "full"
                rows.append({"scope": scope, "align": align, "view": vname, "n_features": len(cols), "u2c_source": src, "u2c_roc": r[src][0], "u2c_pr": r[src][1],
                             "u2c_roc_official": r["official"][0], "u2c_roc_full": r["full"][0], "c2u_roc": r["c2u"][0], "c2u_pr": r["c2u"][1], "mean_roc": (r[src][0] + r["c2u"][0]) / 2})
                log(scope, align, vname, "u2c", round(r[src][0], 3), src, "c2u", round(r["c2u"][0], 3))
                pd.DataFrame(rows).to_csv(OUT / "dev_grid.csv", index=False)
    grid = pd.DataFrame(rows)
    selected = {}
    for scope in ["all", "shared"]:
        b = grid[grid.scope == scope].sort_values("mean_roc", ascending=False).iloc[0]
        cols = views[b.view]
        base = b.mean_roc
        log("LOFO", scope, b.view, b["align"], "base", round(base, 3))
        src_x, src_y = (Xu, yu) if b.u2c_source == "official" else (Xr, yr)
        def quick(use):
            S, Sy = subset(src_x, src_y, scope); mask = c_dev if scope == "all" else c_dev & cic_s
            a = roc_auc_score(yc[mask], fit_predict(b["align"], S, Sy, src_x, Xc, [Xc[mask]], use, n=150_000, trees=100)[0])
            S, Sy = subset(Xc, yc, scope); mask = u_dev if scope == "all" else u_dev & uns_s
            return (a + roc_auc_score(yu[mask], fit_predict(b["align"], S, Sy, Xc, Xr, [Xu[mask]], use, n=150_000, trees=100)[0])) / 2
        ref = quick(cols); lofo = []
        for f in cols:
            v = quick([c for c in cols if c != f]); lofo.append({"scope": scope, "feature": f, "mean_roc_without": v, "change": v - ref})
        L = pd.DataFrame(lofo).sort_values("change", ascending=False)
        L.to_csv(OUT / f"leave_one_feature_out_{scope}.csv", index=False)
        pruned = [c for c in cols if c not in set(L[L.change > 0.005].feature)]
        selected[scope] = {"align": b["align"], "view": b.view, "cols": cols, "u2c_source": b.u2c_source, "dev_mean_roc": float(base),
                           "lofo_cols": pruned, "lofo_dropped": sorted(set(cols) - set(pruned)), "lofo_dev_mean_roc": quick(pruned), "dev_mean_roc_quick": ref}
        log("LOFO done", scope, "dropped", selected[scope]["lofo_dropped"], round(ref, 3), "->", round(selected[scope]["lofo_dev_mean_roc"], 3))
    json.dump(selected, open(OUT / "selected.json", "w"), indent=1)
    log("done")

if MODE == "final":
    sel = json.load(open(OUT / "selected.json"))
    c_fin, u_fin = ~c_dev, ~u_dev
    r_fin = raw["part"].isin([3, 4]).values
    rows, fam_rows = [], []

    def otsu(p):
        h, e = np.histogram(p, bins=200)
        h = h / h.sum(); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
        return float(c[int(np.argmax((mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)))])

    def at(y, s, t):
        h = s >= t
        return {"f1": f1_score(y, h, zero_division=0), "recall": float(h[y == 1].mean()), "fpr": float(h[y == 0].mean())}

    def best_t(y, s):
        qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
        return float(qs[int(np.argmax([f1_score(y, s >= t, zero_division=0) for t in qs]))])

    def record(exp, direction, ev, y, s):
        base = {"experiment": exp, "direction": direction, "eval": ev, "n": len(y), "attack_rate": float(y.mean()), "roc_auc": roc_auc_score(y, s),
                "pr_auc": average_precision_score(y, s), "flag_all_f1": 2 * y.mean() / (1 + y.mean())}
        rows.append({**base, "threshold_rule": "unlabeled (Otsu)", **at(y, s, otsu(s))})
        rows.append({**base, "threshold_rule": "true attack rate known", **at(y, s, float(np.quantile(s, 1 - y.mean())))})
        a, b = [], []
        for seed in (1, 2, 3):
            i = np.random.default_rng(seed).choice(len(y), 500, replace=False)
            a.append(at(y, s, float(np.quantile(s, 1 - y[i].mean())))); b.append(at(y, s, best_t(y[i], s[i])))
        rows.append({**base, "threshold_rule": "attack rate from 500 labelled flows", **pd.DataFrame(a).mean().to_dict()})
        rows.append({**base, "threshold_rule": "best F1 on 500 labelled flows", **pd.DataFrame(b).mean().to_dict()})
        log(exp, direction, ev, "roc", round(base["roc_auc"], 3), "pr", round(base["pr_auc"], 3), [(r["threshold_rule"][:12], round(r["f1"], 3), round(r["fpr"], 2)) for r in rows[-4:]])

    def run_scope(exp, scope, cols, align, u2c_src):
        Xs, ys = (Xu, yu) if u2c_src == "official" else (Xr, yr)
        S, Sy = subset(Xs, ys, scope)
        m = c_fin if scope == "all" else c_fin & cic_s
        p = fit_predict(align, S, Sy, Xs, Xc, [Xc[m]], cols)[0]
        record(exp, "UNSW->CICIDS", "Thu-Fri", yc[m], p)
        names = cic.loc[m, "label_name"].values
        t = otsu(p)
        for f in sorted(set(names[yc[m] == 1])):
            fam_rows.append({"experiment": exp, "direction": "UNSW->CICIDS", "family": f, "n": int((names == f).sum()), "recall": float((p[names == f] >= t).mean()),
                             "roc_vs_benign": roc_auc_score(np.r_[np.ones((names == f).sum()), np.zeros((yc[m] == 0).sum())], np.r_[p[names == f], p[yc[m] == 0]])})
        S, Sy = subset(Xc, yc, scope)
        m1 = u_fin if scope == "all" else u_fin & uns_s
        m2 = r_fin if scope == "all" else r_fin & raw_s
        p1, p2 = fit_predict(align, S, Sy, Xc, Xr, [Xu[m1], Xr[m2]], cols)
        record(exp, "CICIDS->UNSW", "official test file", yu[m1], p1)
        record(exp, "CICIDS->UNSW", "full capture parts 3-4", yr[m2], p2)
        names = unsw.loc[m1, "attack_cat"].astype(str).str.strip().values
        t = otsu(p1)
        for f in sorted(set(names[yu[m1] == 1])):
            fam_rows.append({"experiment": exp, "direction": "CICIDS->UNSW", "family": f, "n": int((names == f).sum()), "recall": float((p1[names == f] >= t).mean()),
                             "roc_vs_benign": roc_auc_score(np.r_[np.ones((names == f).sum()), np.zeros((yu[m1] == 0).sum())], np.r_[p1[names == f], p1[yu[m1] == 0]])})

    run_scope("all attacks, locked config", "all", sel["all"]["cols"], sel["all"]["align"], sel["all"]["u2c_source"])
    run_scope("shared attacks only, locked config", "shared", sel["shared"]["cols"], sel["shared"]["align"], sel["shared"]["u2c_source"])
    run_scope("shared attacks, features chosen with dev labels", "shared", sel["shared"]["lofo_cols"], sel["shared"]["align"], sel["shared"]["u2c_source"])
    pd.DataFrame(rows).to_csv(OUT / "final_results.csv", index=False)
    pd.DataFrame(fam_rows).to_csv(OUT / "final_by_family.csv", index=False)

    # Benign-only anomaly detection: no attack labels from either dataset are used for fitting.
    cols = VIEWS["counts+rates+ratios"]
    an = []

    def anomaly_models(Xb):
        Q = Xb.fillna(0).values
        iso = IsolationForest(n_estimators=200, max_samples=4096, n_jobs=8, random_state=SEED).fit(Q)
        ae = MLPRegressor(hidden_layer_sizes=(16, 8, 16), batch_size=2048, max_iter=30, early_stopping=True, random_state=SEED).fit(Q, Q)
        return {"Isolation Forest": lambda X: -iso.score_samples(X.fillna(0).values), "Autoencoder": lambda X: ((ae.predict(X.fillna(0).values) - X.fillna(0).values) ** 2).mean(1)}

    def score_anomaly(name, models, Y_sets):
        for mname, f in models.items():
            for split, X, y in Y_sets:
                s = f(X)
                an.append({"method": mname, "trained_on": name, "split": split, "n": len(y), "attack_rate": float(y.mean()), "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), **at(y, s, otsu(s))})
                log(mname, name, split, "roc", round(an[-1]["roc_auc"], 3), "f1", round(an[-1]["f1"], 3), "fpr", round(an[-1]["fpr"], 2))

    def benign(X, y, n=200_000):
        b = np.where(y == 0)[0]
        return X.iloc[np.random.default_rng(SEED).choice(b, min(n, len(b)), replace=False)]

    qc = lambda X: quantile(X, Xc, cols)
    qr = lambda X: quantile(X, Xr, cols)
    score_anomaly("UNSW benign, tested on CICIDS", anomaly_models(qr(benign(Xr, yr))), [("dev", qc(Xc[c_dev]), yc[c_dev]), ("final", qc(Xc[c_fin]), yc[c_fin])])
    score_anomaly("CICIDS benign, tested on UNSW", anomaly_models(qc(benign(Xc, yc))), [("dev", qr(Xu[u_dev]), yu[u_dev]), ("final", qr(Xu[u_fin]), yu[u_fin])])
    monday = (cic["capture"] == "Monday-WorkingHours").values
    tue_wed = c_dev & ~monday
    score_anomaly("CICIDS Monday (target's own clean traffic)", anomaly_models(qc(Xc[monday].iloc[::2])), [("dev (Tue-Wed)", qc(Xc[tue_wed]), yc[tue_wed]), ("final", qc(Xc[c_fin]), yc[c_fin])])
    pd.DataFrame(an).to_csv(OUT / "final_anomaly.csv", index=False)

    # Few-shot: source plus labelled target flows, both for random and for earlier-period labels.
    few = []

    def few_scores(Xs, ys, Xpool, Xl, yl, Xte, cols, seed):
        idx = np.random.default_rng(seed).choice(len(Xs), min(200_000, len(Xs)), replace=False)
        imp = SimpleImputer(strategy="median").fit(Xs.iloc[idx][cols])
        sc = StandardScaler().fit(imp.transform(Xs.iloc[idx][cols]))
        z = lambda X: sc.transform(imp.transform(X[cols]))
        Zs = z(Xs.iloc[idx])
        g = coral_map(Zs, z(Xpool)[:: max(1, len(Xpool) // 300000)])
        Xf = np.vstack([g(Zs), z(Xl)])
        w = np.r_[np.ones(len(idx)), np.full(len(Xl), len(idx) / len(Xl))]
        m = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features="sqrt", n_jobs=8, random_state=seed, class_weight="balanced").fit(Xf, np.r_[ys[idx], yl], sample_weight=w)
        return m.predict_proba(z(Xte))[:, 1]

    def one(direction, kind, label, Xs, ys, Xpool, Xl, yl, Xte, yte, seed):
        oof = np.zeros(len(yl))
        for fold in range(3):
            va = (np.arange(len(yl)) % 3) == fold
            oof[va] = few_scores(Xs, ys, Xpool, Xl[~va], yl[~va], Xl[va], cols, seed)
        s = few_scores(Xs, ys, Xpool, Xl, yl, Xte, cols, seed)
        few.append({"direction": direction, "kind": kind, "labelled": label, "seed": seed, "n_test": len(yte), "attack_rate": float(yte.mean()), "roc_auc": roc_auc_score(yte, s), "pr_auc": average_precision_score(yte, s), "flag_all_f1": 2 * yte.mean() / (1 + yte.mean()), **at(yte, s, best_t(yl, oof))})
        log(direction, kind, label, seed, "roc", round(few[-1]["roc_auc"], 3), "f1", round(few[-1]["f1"], 3), "fpr", round(few[-1]["fpr"], 3))

    for direction, Xs, ys, Xpool, Xfin, yfin, Xdev, ydev in [("UNSW->CICIDS", Xu, yu, Xc, Xc[c_fin], yc[c_fin], Xc[c_dev], yc[c_dev]), ("CICIDS->UNSW", Xc, yc, Xr, Xu[u_fin], yu[u_fin], Xu[u_dev], yu[u_dev])]:
        for seed in (1, 2, 3):
            for frac in (0.01, 0.05):
                rs = np.random.default_rng(seed); order = rs.permutation(len(yfin)); k = int(frac * len(yfin))
                lab, rest = order[:k], order[k:]
                one(direction, "random sample of the tested period", f"{int(frac * 100)}%", Xs, ys, Xpool, Xfin.iloc[lab], yfin[lab], Xfin.iloc[rest], yfin[rest], seed)
            for n in (500, 2000):
                lab = np.random.default_rng(seed).choice(len(ydev), n, replace=False)
                one(direction, "earlier period only", n, Xs, ys, Xpool, Xdev.iloc[lab], ydev[lab], Xfin, yfin, seed)
            pd.DataFrame(few).to_csv(OUT / "final_few_shot.csv", index=False)
    log("done")
