"""Zero-label method, version 2. Usage: zero_label_v2.py build|dev|hidden [nf_data_dir] [cic2017_dir]

Rules, fixed before any score was computed:
- No label of the dataset being tested is used for training, alignment, thresholds or choosing a method.
- Features use only source IP, destination IP, destination port, flow start time and whether the flow got a reply,
  so every dataset gets identical features whatever tool recorded it.
- Every dataset is split by alternating clock hours: odd hours are unlabeled calibration, even hours are marked.
- dev     UNSW, ToN, CIC2018 and BoT-IoT. Each one is marked after training on the other three. The locked
          configuration has the highest mean balanced accuracy over the four, ties broken by mean ROC-AUC.
- hidden  CICIDS2017 with IP addresses. Two methods, both locked before this run, are trained on the dev datasets
          and marked once: version 1 (section 9 of the report) and the version 2 configuration from dev.
          The script refuses a second run.
"""
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc_arrow
import pyarrow.parquet as pq
from lightgbm import LGBMClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

warnings.filterwarnings("ignore")
MODE = sys.argv[1]
ROOT = Path(__file__).resolve().parents[1]
NF = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / "nf_data"
CIC17 = Path(sys.argv[3]) if len(sys.argv) > 3 else ROOT.parent / "cic2017_ip"
CACHE = NF / "cache_v2"
OUT = ROOT / "outputs" / "zero_label_v2"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
SEED = 42
KEEP = 3_000_000
NF_FILES = {"UNSW": "NF-UNSW-NB15-v3.parquet", "ToN": "NF-ToN-IoT-v3.parquet", "CIC2018": "NF-CICIDS2018-v3.parquet", "BoT": "NF-BoT-IoT-v3.parquet"}
sys.path.insert(0, str(Path(__file__).parent))
import nf_transfer as N  # noqa: E402  shared flow column list
WINDOWS = (1, 10, 60)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def gid(key):
    return np.unique(key, return_inverse=True)[1].astype("int64")


def distinct(g, other):
    pairs = np.unique((g << 32) | other)
    return np.bincount((pairs >> 32).astype("int64"), minlength=int(g.max()) + 1)[g]


def relative_to_own_baseline(g, owner, value):
    """How unusual a window is for this computer: its value minus that computer's median window."""
    first = np.unique(g, return_index=True)[1]
    per_group = pd.DataFrame({"owner": owner[first], "v": value[first]})
    med = per_group.groupby("owner").v.median()
    return value - med.reindex(owner).values


def computer_features(src, dst, port, minute, noreply):
    out = {}
    for w in WINDOWS:
        b = minute // w
        g = gid((src << 24) | b); f = np.bincount(g)
        out[f"src_flows_{w}"] = np.log1p(f[g])
        out[f"src_dst_ips_{w}"] = np.log1p(distinct(g, dst))
        ports = distinct(g, port)
        out[f"src_dst_ports_{w}"] = np.log1p(ports)
        out[f"src_port_fanout_{w}"] = ports / f[g]
        out[f"src_noreply_frac_{w}"] = np.bincount(g, weights=noreply)[g] / f[g]
        if w == 1:
            out["rel_src_flows"] = relative_to_own_baseline(g, src, out["src_flows_1"])
            out["rel_src_dst_ports"] = relative_to_own_baseline(g, src, out["src_dst_ports_1"])
            out["rel_src_dst_ips"] = relative_to_own_baseline(g, src, out["src_dst_ips_1"])
        g = gid((dst << 24) | b); f = np.bincount(g)
        out[f"dst_flows_{w}"] = np.log1p(f[g])
        out[f"dst_src_ips_{w}"] = np.log1p(distinct(g, src))
        out[f"dst_noreply_frac_{w}"] = np.bincount(g, weights=noreply)[g] / f[g]
        if w == 1:
            out["rel_dst_flows"] = relative_to_own_baseline(g, dst, out["dst_flows_1"])
        g = gid((((src << 20) | dst) << 22) | b); f = np.bincount(g)
        out[f"pair_flows_{w}"] = np.log1p(f[g])
        out[f"pair_dst_ports_{w}"] = np.log1p(distinct(g, port))
        s = gid((g << 16) | port)
        out[f"service_repeat_{w}"] = np.log1p(np.bincount(s)[s])
        g = gid((((src << 16) | port) << 22) | b)
        out[f"same_port_many_hosts_{w}"] = np.log1p(distinct(g, dst))
    return {k: v.astype("float32") for k, v in out.items()}


def codes(values):
    if isinstance(values, pa.Array):
        return pc_arrow.dictionary_encode(values).indices.to_numpy(zero_copy_only=False).astype("int64")
    return pd.factorize(pd.Series(values))[0].astype("int64")


def finish(name, ctx, label, attack, minute, src, keep_rows, dedupe_frame):
    d = pd.DataFrame({k: v[keep_rows] for k, v in ctx.items()})
    d["label"] = label[keep_rows].astype("int8")
    d["attack"] = attack[keep_rows]
    d["src_code"] = src[keep_rows]
    d["win10"] = (minute[keep_rows] // 10).astype("int64")
    hour = minute[keep_rows] // 60
    order = {h: i for i, h in enumerate(sorted(np.unique(hour)))}
    d["is_test"] = pd.Series(hour).map(order).values % 2 == 0
    before = len(d)
    dup = dedupe_frame.duplicated().values
    d = d[~dup].reset_index(drop=True)
    d.to_parquet(CACHE / f"{name}.parquet")
    log(name, "rows", before, "after duplicates", len(d), "benign", int((d.label == 0).sum()), "hours", len(order), "test rows", int(d.is_test.sum()), "test attack rate", round(float(d[d.is_test].label.mean()), 3))


if MODE == "build":
    for name, file in NF_FILES.items():
        if (CACHE / f"{name}.parquet").exists():
            log(name, "already built"); continue
        path = NF / file
        t = pq.read_table(path, columns=["IPV4_SRC_ADDR", "IPV4_DST_ADDR", "L4_DST_PORT", "FLOW_START_MILLISECONDS", "OUT_PKTS", "Label", "Attack"])
        n = t.num_rows
        src, dst = codes(t["IPV4_SRC_ADDR"].combine_chunks()), codes(t["IPV4_DST_ADDR"].combine_chunks())
        port = t["L4_DST_PORT"].to_numpy().astype("int64")
        ms = t["FLOW_START_MILLISECONDS"].to_numpy().astype("int64")
        minute = (ms - ms.min()) // 60_000
        label = t["Label"].to_numpy().astype("int64")
        noreply = (t["OUT_PKTS"].to_numpy() <= 0).astype("float64")
        rng = np.random.default_rng(SEED)
        pick = rng.random(n) < min(1.0, KEEP / n)
        if name == "BoT":  # 0.3% of BoT is benign, so every benign flow is kept and only attacks are sampled.
            pick |= label == 0
        rows = np.where(pick)[0]
        attack = np.full(n, "", dtype=object)
        attack[rows] = np.asarray(t["Attack"].take(pa.array(rows)).to_pandas().astype(str).str.strip().values)
        del t
        ctx = computer_features(src, dst, port, minute, noreply)
        # Duplicates are judged on the same flow columns as section 8, read only for the sampled rows.
        flow, start = [], 0
        for batch in pq.ParquetFile(path).iter_batches(batch_size=500_000, columns=N.RAW + ["Label"]):
            m = pick[start:start + batch.num_rows]
            flow.append(batch.filter(pa.array(m)).to_pandas())
            start += batch.num_rows
        flow = pd.concat(flow, ignore_index=True)
        finish(name, ctx, label, attack, minute, src, rows, flow)

    if not (CACHE / "CICIDS2017.parquet").exists():
        frames = []
        with zipfile.ZipFile(CIC17 / "GeneratedLabelledFlows.zip") as z:
            for member in sorted(m for m in z.namelist() if m.lower().endswith(".csv")):
                with z.open(member) as fh:
                    d = pd.read_csv(fh, encoding="latin-1", low_memory=False)
                d.columns = [c.strip() for c in d.columns]
                d = d.dropna(subset=["Source IP", "Timestamp", "Label"])
                d["file"] = Path(member).name
                frames.append(d)
                log(member, len(d))
        d = pd.concat(frames, ignore_index=True)
        ts = pd.to_datetime(d["Timestamp"].str.strip(), format="mixed", dayfirst=True)
        # The CSVs print afternoon times on a 12-hour clock without AM/PM; the capture ran in working hours only.
        ts = ts + pd.to_timedelta(np.where(ts.dt.hour < 8, 12, 0), unit="h")
        log("hours of day after the 12-hour fix", sorted(ts.dt.hour.unique().tolist()))
        minute = ((ts - ts.min()).dt.total_seconds() // 60).astype("int64").values
        src, dst = codes(d["Source IP"].astype(str).values), codes(d["Destination IP"].astype(str).values)
        port = pd.to_numeric(d["Destination Port"], errors="coerce").fillna(0).astype("int64").values
        noreply = (pd.to_numeric(d["Total Backward Packets"], errors="coerce").fillna(0) <= 0).astype("float64").values
        label_name = d["Label"].astype(str).str.strip()
        label = (label_name.str.upper() != "BENIGN").astype("int64").values
        ctx = computer_features(src, dst, port, minute, noreply)
        flow_cols = [c for c in d.columns if c not in ("Flow ID", "Source IP", "Source Port", "Destination IP", "Timestamp", "file")]
        counts = pd.DataFrame({"file": d["file"], "label": label_name}).value_counts().rename("rows").reset_index()
        counts.to_csv(OUT / "cicids2017_rows_by_file.csv", index=False)
        finish("CICIDS2017", ctx, label, label_name.values, minute, src, np.arange(len(d)), d[flow_cols].reset_index(drop=True))
    log("done")
    sys.exit()

ALL = [c for c in pd.read_parquet(CACHE / "UNSW.parquet").columns if c not in ("label", "attack", "is_test", "src_code", "win10")]
V1 = ["src_flows_1", "src_dst_ips_1", "src_dst_ports_1", "src_port_fanout_1", "src_noreply_frac_1", "dst_flows_1", "dst_src_ips_1", "dst_noreply_frac_1", "pair_flows_1", "pair_dst_ports_1", "service_repeat_1"]
RELATIVE = [c for c in ALL if c.startswith("rel_") or "_frac_" in c or "_fanout_" in c]
VIEWS = {"version 1 (1 minute)": V1, "three windows": [c for c in ALL if not c.startswith("rel_")], "relative only": RELATIVE, "three windows + relative": ALL}
VARIANTS = ["model", "model + computer smoothing", "model + own-network detector", "model + detector + smoothing", "own-network detector only"]
rng = np.random.default_rng(SEED)


def load(name):
    d = pd.read_parquet(CACHE / f"{name}.parquet")
    d[ALL] = d[ALL].replace([np.inf, -np.inf], np.nan)
    return d


def cap(idx, n):
    return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))


def otsu(p):
    h, e = np.histogram(p, bins=200)
    h = h / h.sum(); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
    return float(c[int(np.argmax((mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)))])


def lgbm():
    return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=10, random_state=SEED, verbose=-1)


def balanced_weights(y, tag):
    w = np.zeros(len(y))
    for s in np.unique(tag):
        for c in (0, 1):
            m = (tag == s) & (y == c)
            if m.sum():
                w[m] = 1.0 / m.sum()
    return w / w.mean()


def consistent_features(sources, cols):
    """Keep features that point the same way, clearly, in every source dataset."""
    keep = []
    for c in cols:
        a = []
        for S in sources:
            i = cap(np.arange(len(S)), 200_000)
            v = S[c].values[i]; v = np.where(np.isnan(v), np.nanmedian(v), v)
            a.append(roc_auc_score(S.label.values[i], v) if len(np.unique(S.label.values[i])) == 2 else 0.5)
        a = np.array(a)
        if (a > 0.55).all() or (a < 0.45).all():
            keep.append(c)
    return keep if len(keep) >= 3 else list(cols)


def smooth(score, frame):
    g = pd.Series(score).groupby([frame.src_code.values, frame.win10.values]).transform("mean").values
    return 0.5 * score + 0.5 * g


def unit(cal, x):
    lo, hi = np.quantile(cal, [0.01, 0.99])
    return np.clip((x - lo) / max(hi - lo, 1e-9), 0, 1)


def scores(sources, target, cols, selftrain):
    parts, ys, tags = [], [], []
    for i, S in enumerate(sources):
        idx = cap(np.arange(len(S)), 200_000)
        parts.append(S.iloc[idx][cols]); ys.append(S.label.values[idx]); tags.append(np.full(len(idx), i))
    X, y, tag = pd.concat(parts, ignore_index=True), np.concatenate(ys), np.concatenate(tags)
    cal, test = target.iloc[cap(np.where(~target.is_test.values)[0], 300_000)], target.iloc[cap(np.where(target.is_test.values)[0], 1_000_000)]
    m = lgbm().fit(X, y, sample_weight=balanced_weights(y, tag))
    pc = m.predict_proba(cal[cols])[:, 1]
    if selftrain:
        t = otsu(pc); high, low = pc >= t, pc < t
        if high.sum() >= 200 and low.sum() >= 200:
            a = cap(np.where(pc >= np.median(pc[high]))[0], 100_000)
            b = cap(np.where(pc <= np.median(pc[low]))[0], 100_000)
            X2 = pd.concat([X, cal[cols].iloc[np.r_[a, b]]], ignore_index=True)
            y2, tag2 = np.r_[y, np.ones(len(a)), np.zeros(len(b))].astype(int), np.r_[tag, np.full(len(a) + len(b), len(sources))]
            m = lgbm().fit(X2, y2, sample_weight=balanced_weights(y2, tag2))
            pc = m.predict_proba(cal[cols])[:, 1]
    pt = m.predict_proba(test[cols])[:, 1]
    fill = cal[cols].median()
    iso = IsolationForest(n_estimators=200, max_samples=4096, n_jobs=10, random_state=SEED).fit(cal[cols].fillna(fill).values)
    ic, it = -iso.score_samples(cal[cols].fillna(fill).values), -iso.score_samples(test[cols].fillna(fill).values)
    ic, it = unit(ic, ic), unit(ic, it)
    out = {"model": (pc, pt), "own-network detector only": (ic, it)}
    out["model + computer smoothing"] = (smooth(pc, cal), smooth(pt, test))
    out["model + own-network detector"] = ((pc + ic) / 2, (pt + it) / 2)
    out["model + detector + smoothing"] = (smooth((pc + ic) / 2, cal), smooth((pt + it) / 2, test))
    return out, test


def marks(y, s, t):
    h = s >= t
    rec, fpr = float(h[y == 1].mean()), float(h[y == 0].mean())
    return {"balanced_accuracy": (rec + 1 - fpr) / 2, "accuracy": float((h == (y == 1)).mean()), "roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s),
            "f1": f1_score(y, h, zero_division=0), "recall": rec, "fpr": fpr, "flag_all_f1": 2 * y.mean() / (1 + y.mean()), "attack_rate": float(y.mean()), "n": len(y)}


def evaluate(tags, sources, target, view, invariant, selftrain, fam=None):
    cols = consistent_features(sources, VIEWS[view]) if invariant else VIEWS[view]
    rows = []
    sc, test = scores(sources, target, cols, selftrain)
    y = test.label.values
    for variant, (pc, pt) in sc.items():
        t = otsu(pc)
        rows.append({**tags, "view": view, "consistent_features_only": invariant, "self_training": selftrain, "variant": variant, "n_features": len(cols), **marks(y, pt, t)})
        if fam is not None and variant == fam[0]:
            names = test.attack.values
            for f in sorted(set(names[y == 1])):
                fam[1].append({**tags, "family": f, "n": int((names == f).sum()), "recall": float((pt[names == f] >= t).mean())})
    log(tags, view, invariant, selftrain, [(r["variant"][:22], round(r["balanced_accuracy"], 3), round(r["roc_auc"], 3)) for r in rows])
    return rows, cols


if MODE == "dev":
    if (OUT / "hidden_results.csv").exists():
        sys.exit("the hidden test was already scored, dev must not change after it")
    data = {k: load(k) for k in NF_FILES}
    rows = []
    for view in VIEWS:
        for invariant in (False, True):
            for selftrain in (False, True):
                for tgt in NF_FILES:
                    r, _ = evaluate({"target": tgt, "sources": "+".join(k for k in NF_FILES if k != tgt)}, [data[k] for k in NF_FILES if k != tgt], data[tgt], view, invariant, selftrain)
                    rows += r
                    pd.DataFrame(rows).to_csv(OUT / "dev_all.csv", index=False)
    grid = pd.DataFrame(rows)
    keys = ["view", "consistent_features_only", "self_training", "variant"]
    mean = grid.groupby(keys)[["balanced_accuracy", "roc_auc"]].mean().sort_values(["balanced_accuracy", "roc_auc"], ascending=False)
    mean.to_csv(OUT / "dev_ranking.csv")
    best = dict(zip(keys, mean.index[0]))
    best = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else v) for k, v in best.items()}
    json.dump({**best, "dev_mean_balanced_accuracy": float(mean.iloc[0].balanced_accuracy), "dev_mean_roc_auc": float(mean.iloc[0].roc_auc)}, open(OUT / "selected.json", "w"), indent=1)
    log("locked", best, mean.iloc[0].to_dict())

if MODE == "hidden":
    done = OUT / "hidden_results.csv"
    if done.exists():
        sys.exit("the hidden test was already scored once:\n" + done.read_text())
    sel = json.load(open(OUT / "selected.json"))
    data = {k: load(k) for k in NF_FILES}
    target = load("CICIDS2017")
    rows, fams = [], []
    v1_sources = ["UNSW", "ToN", "CIC2018"]
    r, _ = evaluate({"method": "version 1, locked in section 9", "sources": "+".join(v1_sources)}, [data[k] for k in v1_sources], target, "version 1 (1 minute)", False, True, ("model", fams))
    rows += [x for x in r if x["variant"] == "model"]
    r, cols = evaluate({"method": "version 2, locked on dev", "sources": "+".join(NF_FILES)}, list(data.values()), target, sel["view"], sel["consistent_features_only"], sel["self_training"], (sel["variant"], fams))
    rows += [x for x in r if x["variant"] == sel["variant"]]
    pd.DataFrame(rows).to_csv(done, index=False)
    pd.DataFrame(fams).to_csv(OUT / "hidden_by_family.csv", index=False)
    json.dump({"version_2_features": cols}, open(OUT / "hidden_features_used.json", "w"), indent=1)
    log("done")
