"""Zero-label experiments on NetFlow v3. Usage: nf_zero_label.py context|dev|holdout [data_dir]

Rules, fixed before any score was computed:
- No target label is used for training, alignment, thresholds or choosing a method. Target labels only mark.
- context  adds per-computer behaviour in one-minute windows, computed on the whole file before sampling.
- dev      UNSW, ToN and CIC2018 only. Every configuration is trained on source datasets and marked on the other.
           The score for a target comes from the configuration with the best mean ROC-AUC between the other two.
           selected.json locks the configuration with the best mean ROC-AUC over the three leave-one-out tests.
- holdout  trains the locked configuration on all three and marks BoT-IoT, which no earlier step reads. It runs once.
"""
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc_arrow
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

MODE = sys.argv[1]
sys.path.insert(0, str(Path(__file__).parent))
sys.argv[1] = "library"
import nf_transfer as N

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / "nf_data"
CACHE = DATA / "cache_context"
OUT = ROOT / "outputs" / "netflow_v3" / "zero_label"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
SEED = 42
DEV_FILES = dict(N.FILES)
ALL_FILES = {**DEV_FILES, "BoT": "NF-BoT-IoT-v3.parquet"}
CONTEXT = ["src_flows", "src_dst_ips", "src_dst_ports", "src_port_fanout", "src_noreply_frac", "dst_flows", "dst_src_ips", "dst_noreply_frac", "pair_flows", "pair_dst_ports", "service_repeat"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def group_ids(key):
    return np.unique(key, return_inverse=True)[1]


def distinct_per_group(g, other):
    pairs = np.unique((g.astype("int64") << 32) | other.astype("int64"))
    return np.bincount((pairs >> 32).astype("int64"), minlength=int(g.max()) + 1)[g]


def window_context(path, names):
    t = pq.read_table(path, columns=[names[c] for c in ("IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "FLOW_START_MILLISECONDS", "OUT_PKTS")])
    def codes(c):
        return pc_arrow.dictionary_encode(t[names[c]].combine_chunks()).indices.to_numpy(zero_copy_only=False).astype("int64")
    src, dst = codes("IPV4_SRC_ADDR"), codes("IPV4_DST_ADDR")
    port = t[names["L4_DST_PORT"]].to_numpy().astype("int64")
    ms = t[names["FLOW_START_MILLISECONDS"]].to_numpy().astype("int64")
    noreply = (t[names["OUT_PKTS"]].to_numpy() <= 0).astype("float64")
    del t
    minute = (ms - ms.min()) // 60_000
    del ms
    out = {}
    g = group_ids((src << 24) | minute)
    flows = np.bincount(g)
    out["src_flows"] = flows[g]
    out["src_dst_ips"] = distinct_per_group(g, dst)
    out["src_dst_ports"] = distinct_per_group(g, port)
    out["src_port_fanout"] = out["src_dst_ports"] / flows[g]
    out["src_noreply_frac"] = np.bincount(g, weights=noreply)[g] / flows[g]
    g = group_ids((dst << 24) | minute)
    flows = np.bincount(g)
    out["dst_flows"] = flows[g]
    out["dst_src_ips"] = distinct_per_group(g, src)
    out["dst_noreply_frac"] = np.bincount(g, weights=noreply)[g] / flows[g]
    g = group_ids((((src << 20) | dst) << 22) | minute)
    flows = np.bincount(g)
    out["pair_flows"] = flows[g]
    out["pair_dst_ports"] = distinct_per_group(g, port)
    out["service_repeat"] = np.bincount(group_ids((g.astype("int64") << 16) | port))[group_ids((g.astype("int64") << 16) | port)]
    for k in list(out):
        if not k.endswith(("_frac", "_fanout")):
            out[k] = np.log1p(out[k])
    return {k: out[k].astype("float32") for k in CONTEXT}


if MODE == "context":
    for name, file in ALL_FILES.items():
        path = DATA / file
        target = CACHE / f"{name}.parquet"
        if target.exists():
            log(name, "already built")
            continue
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        names = {c.strip(): c for c in pf.schema_arrow.names}
        ctx = window_context(path, names)
        log(name, "context built", n)
        rng, keep, parts, start = np.random.default_rng(SEED), min(1.0, N.KEEP / n), [], 0
        columns = [names[c] for c in N.RAW + ["Label", "Attack", "FLOW_START_MILLISECONDS"]]
        for batch in pf.iter_batches(batch_size=500_000, columns=columns):
            pick = rng.random(batch.num_rows) < keep
            if name == "BoT":  # 0.3% of BoT is benign, so every benign flow is kept and only attacks are sampled.
                pick = pick | (batch.column(columns.index(names["Label"])).to_numpy(zero_copy_only=False) == 0)
            where = np.where(pick)[0]
            d = batch.filter(pa.array(pick)).to_pandas()
            d.columns = [c.strip() for c in d.columns]
            for c in N.RAW:
                d[c] = d[c].astype("float64")
            for c in CONTEXT:
                d[c] = ctx[c][start + where]
            parts.append(d)
            start += batch.num_rows
        d = pd.concat(parts, ignore_index=True)
        before = len(d)
        d = d.drop_duplicates(subset=N.RAW + ["Label"], ignore_index=True)
        X = N.build(d)
        for c in CONTEXT:
            X[c] = d[c].values
        hour = (d["FLOW_START_MILLISECONDS"].astype("float64") // 3_600_000).astype("int64")
        order = {h: i for i, h in enumerate(sorted(hour.unique()))}
        X["label"] = d["Label"].astype("int8").values
        X["attack"] = d["Attack"].astype(str).str.strip().values
        X["is_test"] = hour.map(order).values % 2 == 0
        X.to_parquet(target)
        log(name, "sampled", before, "after duplicates", len(X), "benign", int((X.label == 0).sum()), "test rows", int(X.is_test.sum()))
    log("done")
    sys.exit()

if MODE not in ("dev", "holdout"):
    sys.exit("mode must be context, dev or holdout")
rng = np.random.default_rng(SEED)


def load(names):
    out = {}
    for k in names:
        d = pd.read_parquet(CACHE / f"{k}.parquet")
        cols = [c for c in d.columns if c not in ("label", "attack", "is_test")]
        d[cols] = d[cols].replace([np.inf, -np.inf], np.nan)
        out[k] = d
    return out


def cap(idx, n):
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def otsu(p):
    h, e = np.histogram(p, bins=200)
    h = h / h.sum(); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
    return float(c[int(np.argmax((mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)))])


def quantile(X, pool, cols):
    X = X[cols].copy()
    for k in cols:
        if k in N.BINARY:
            continue
        ref = np.sort(pool[k].dropna().values[:: max(1, len(pool) // 200000)])
        X[k] = (np.searchsorted(ref, X[k].values, side="right") / len(ref)).astype("float32")
    return X


def lgbm():
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=10, random_state=SEED, verbose=-1)


def views(d):
    flow = [c for c in d.columns if c not in ("label", "attack", "is_test") and c not in N.WIDE_ONLY and c not in CONTEXT]
    return {"per-flow": flow, "per-computer": CONTEXT, "per-flow + per-computer": flow + CONTEXT}


def balanced_weights(y, source):
    """Each source counts equally, and inside it attack and normal count equally."""
    w = np.zeros(len(y))
    for s in np.unique(source):
        for c in (0, 1):
            m = (source == s) & (y == c)
            if m.sum():
                w[m] = 1.0 / m.sum()
    return w / w.mean()


def fit_score(sources, target, cols, align, selftrain):
    parts, ys, tags = [], [], []
    for i, S in enumerate(sources):
        idx = cap(np.arange(len(S)), 200_000)
        parts.append(quantile(S.iloc[idx], S, cols) if align == "quantile" else S.iloc[idx][cols])
        ys.append(S.label.values[idx]); tags.append(np.full(len(idx), i))
    X, y, tag = pd.concat(parts, ignore_index=True), np.concatenate(ys), np.concatenate(tags)
    cal_idx = cap(np.where(~target.is_test.values)[0], 300_000)
    test_idx = cap(np.where(target.is_test.values)[0], 1_000_000)
    pool = target.iloc[cal_idx]
    prep = (lambda f: quantile(f, pool, cols)) if align == "quantile" else (lambda f: f[cols])
    Xcal, Xtest = prep(pool), prep(target.iloc[test_idx])
    m = lgbm().fit(X, y, sample_weight=balanced_weights(y, tag))
    pcal = m.predict_proba(Xcal)[:, 1]
    if selftrain:
        t = otsu(pcal)
        high, low = pcal >= t, pcal < t
        if high.sum() >= 200 and low.sum() >= 200:
            confident_attack = np.where(pcal >= np.median(pcal[high]))[0]
            confident_normal = np.where(pcal <= np.median(pcal[low]))[0]
            take = np.r_[cap(confident_attack, 100_000), cap(confident_normal, 100_000)]
            pseudo_y = np.r_[np.ones(min(len(confident_attack), 100_000)), np.zeros(min(len(confident_normal), 100_000))].astype(int)
            X2 = pd.concat([X, Xcal.iloc[take]], ignore_index=True)
            y2, tag2 = np.r_[y, pseudo_y], np.r_[tag, np.full(len(take), len(sources))]
            m = lgbm().fit(X2, y2, sample_weight=balanced_weights(y2, tag2))
            pcal = m.predict_proba(Xcal)[:, 1]
    return pcal, m.predict_proba(Xtest)[:, 1], test_idx


def marks(y, s, t):
    h = s >= t
    qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
    return {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), "f1": f1_score(y, h, zero_division=0), "recall": float(h[y == 1].mean()), "fpr": float(h[y == 0].mean()),
            "flag_all_f1": 2 * y.mean() / (1 + y.mean()), "best_possible_f1": max(f1_score(y, s >= q, zero_division=0) for q in qs), "attack_rate": float(y.mean()), "n": len(y)}


def families(tags, target, idx, s, t):
    names, y = target.attack.values[idx], target.label.values[idx]
    ben = s[y == 0]
    return [{**tags, "family": f, "n": int((names == f).sum()), "recall": float((s[names == f] >= t).mean()), "roc_vs_benign": roc_auc_score(np.r_[np.ones((names == f).sum()), np.zeros(len(ben))], np.r_[s[names == f], ben])} for f in sorted(set(names[y == 1]))]


def run(tags, sources, target, cols, align, selftrain, fam_out=None):
    pc, pt, idx = fit_score(sources, target, cols, align, selftrain)
    t = otsu(pc)
    r = {**tags, "align": align, "self_training": selftrain, **marks(target.label.values[idx], pt, t)}
    if fam_out is not None:
        fam_out += families({k: tags[k] for k in tags}, target, idx, pt, t)
    log(tags, align, selftrain, "roc", round(r["roc_auc"], 3), "f1", round(r["f1"], 3), "fpr", round(r["fpr"], 3), "flag-all", round(r["flag_all_f1"], 3))
    return r


def isolation(target, cols):
    cal, test = cap(np.where(~target.is_test.values)[0], 300_000), cap(np.where(target.is_test.values)[0], 1_000_000)
    pool = target.iloc[cal]
    q = lambda f: quantile(f, pool, cols).fillna(1.0).values
    m = IsolationForest(n_estimators=200, max_samples=4096, n_jobs=10, random_state=SEED).fit(q(pool))
    pc, pt = -m.score_samples(q(pool)), -m.score_samples(q(target.iloc[test]))
    return marks(target.label.values[test], pt, otsu(pc))


CONFIGS = [(v, a, st) for v in ("per-flow", "per-computer", "per-flow + per-computer") for a in ("none", "quantile") for st in (False, True)]

if MODE == "dev":
    if (OUT / "holdout_result.json").exists():
        sys.exit("holdout has already been scored, dev must not change after it")
    data = load(DEV_FILES)
    V = views(next(iter(data.values())))
    rows, fams = [], []
    for v, a, st in CONFIGS:
        for tgt in DEV_FILES:
            others = [k for k in DEV_FILES if k != tgt]
            for src in others:
                rows.append(run({"kind": "one source", "source": src, "target": tgt, "view": v}, [data[src]], data[tgt], V[v], a, st))
            rows.append(run({"kind": "two sources", "source": "+".join(others), "target": tgt, "view": v}, [data[k] for k in others], data[tgt], V[v], a, st, fams))
            pd.DataFrame(rows).to_csv(OUT / "dev_all.csv", index=False)
            pd.DataFrame(fams).to_csv(OUT / "dev_by_family.csv", index=False)
    grid = pd.DataFrame(rows)
    nested = []
    for tgt in DEV_FILES:
        others = [k for k in DEV_FILES if k != tgt]
        inner = grid[(grid.kind == "one source") & grid.source.isin(others) & grid.target.isin(others)]
        best = inner.groupby(["view", "align", "self_training"]).roc_auc.mean().sort_values(ascending=False)
        v, a, st = best.index[0]
        pick = grid[(grid.kind == "two sources") & (grid.target == tgt) & (grid.view == v) & (grid["align"] == a) & (grid.self_training == st)].iloc[0].to_dict()
        nested.append({**pick, "chosen_on_other_datasets_mean_roc": float(best.iloc[0])})
    pd.DataFrame(nested).to_csv(OUT / "dev_zero_label_results.csv", index=False)
    ref = grid[grid.kind == "two sources"].groupby(["view", "align", "self_training"]).roc_auc.mean().sort_values(ascending=False)
    v, a, st = ref.index[0]
    json.dump({"view": v, "align": a, "self_training": bool(st), "dev_mean_roc": float(ref.iloc[0])}, open(OUT / "selected.json", "w"), indent=1)
    iso = []
    for tgt in DEV_FILES:
        for v, cols in V.items():
            iso.append({"target": tgt, "view": v, **isolation(data[tgt], cols)})
            log("isolation", tgt, v, round(iso[-1]["roc_auc"], 3), round(iso[-1]["f1"], 3), round(iso[-1]["fpr"], 3))
    pd.DataFrame(iso).to_csv(OUT / "dev_isolation_forest.csv", index=False)
    log("done")

if MODE == "holdout":
    done = OUT / "holdout_result.json"
    if done.exists():
        sys.exit("holdout was already scored once: " + done.read_text())
    sel = json.load(open(OUT / "selected.json"))
    data = load(ALL_FILES)
    V = views(data["BoT"])
    fams = []
    r = run({"kind": "three sources", "source": "UNSW+ToN+CIC2018", "target": "BoT", "view": sel["view"]}, [data[k] for k in DEV_FILES], data["BoT"], V[sel["view"]], sel["align"], sel["self_training"], fams)
    json.dump(r, open(done, "w"), indent=1, default=float)
    pd.DataFrame(fams).to_csv(OUT / "holdout_by_family.csv", index=False)
    log("done")
