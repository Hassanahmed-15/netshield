"""No-label transfer test on the NetFlow v3 datasets. Usage: nf_transfer.py prep|run [data_dir]

Rules, fixed before any score was computed:
- Every target's flows are split by alternating clock hours. Odd hours are the unlabeled calibration pool,
  even hours are the test rows.
- Target labels are used only to score the test rows. Alignment and the Otsu threshold use all calibration
  traffic, with no label filter, including in the shared-attack test.
- Method choice for a target uses only the two other datasets: highest mean ROC-AUC over the two transfer
  directions between them.
prep   verifies the files against the official counts, samples, removes duplicates, writes cache.
run    scores every configuration in every direction and writes the locked results.
"""
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
MODE = sys.argv[1]
ROOT = Path(__file__).resolve().parents[1]
DATA = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT.parent / "nf_data"
CACHE = DATA / "cache"
OUT = ROOT / "outputs" / "netflow_v3"
OUT.mkdir(parents=True, exist_ok=True)
CACHE.mkdir(parents=True, exist_ok=True)
SEED = 42
KEEP = 3_000_000
FILES = {"UNSW": "NF-UNSW-NB15-v3.parquet", "ToN": "NF-ToN-IoT-v3.parquet", "CIC2018": "NF-CICIDS2018-v3.parquet"}
OFFICIAL_ROWS = {"UNSW": 2_365_424, "ToN": 27_520_260, "CIC2018": 20_115_529}
COUNT = ["IN_BYTES", "OUT_BYTES", "IN_PKTS", "OUT_PKTS"]
DUR = ["FLOW_DURATION_MILLISECONDS", "DURATION_IN", "DURATION_OUT"]
SIZE = ["LONGEST_FLOW_PKT", "SHORTEST_FLOW_PKT", "MIN_IP_PKT_LEN", "MAX_IP_PKT_LEN"]
RATE = ["SRC_TO_DST_SECOND_BYTES", "DST_TO_SRC_SECOND_BYTES", "SRC_TO_DST_AVG_THROUGHPUT", "DST_TO_SRC_AVG_THROUGHPUT"]
IAT = [f"{d}_IAT_{s}" for d in ("SRC_TO_DST", "DST_TO_SRC") for s in ("MIN", "MAX", "AVG", "STDDEV")]
RETR = ["RETRANSMITTED_IN_PKTS", "RETRANSMITTED_OUT_PKTS"]
HIST = ["NUM_PKTS_UP_TO_128_BYTES", "NUM_PKTS_128_TO_256_BYTES", "NUM_PKTS_256_TO_512_BYTES", "NUM_PKTS_512_TO_1024_BYTES", "NUM_PKTS_1024_TO_1514_BYTES"]
EXTRA = ["PROTOCOL", "CLIENT_TCP_FLAGS", "SERVER_TCP_FLAGS", "MIN_TTL", "MAX_TTL", "TCP_WIN_MAX_IN", "TCP_WIN_MAX_OUT"]
RAW = COUNT + DUR + SIZE + RATE + IAT + RETR + HIST + EXTRA
FLAGS = {"fin": 1, "syn": 2, "rst": 4, "psh": 8, "ack": 16}


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def build(d):
    out = {}
    for c in COUNT + DUR + SIZE + RATE + IAT:
        out["log_" + c] = np.log1p(d[c].clip(lower=0))
    ip, op = d.IN_PKTS.clip(lower=1), d.OUT_PKTS.clip(lower=1)
    out["bytes_per_pkt_in"] = np.log1p(d.IN_BYTES.clip(lower=0) / ip)
    out["bytes_per_pkt_out"] = np.log1p(d.OUT_BYTES.clip(lower=0) / op)
    out["byte_ratio"] = out["log_OUT_BYTES"] - out["log_IN_BYTES"]
    out["pkt_ratio"] = out["log_OUT_PKTS"] - out["log_IN_PKTS"]
    out["retr_frac_in"] = d.RETRANSMITTED_IN_PKTS.clip(lower=0) / ip
    out["retr_frac_out"] = d.RETRANSMITTED_OUT_PKTS.clip(lower=0) / op
    total = d[HIST].clip(lower=0).sum(axis=1).clip(lower=1)
    for c in HIST:
        out["frac_" + c] = d[c].clip(lower=0) / total
    out["has_reply"] = (d.OUT_PKTS > 0).astype("float64")
    for name, value in [("tcp", 6), ("udp", 17), ("icmp", 1)]:
        out["is_" + name] = (d.PROTOCOL == value).astype("float64")
    for side, col in [("c", "CLIENT_TCP_FLAGS"), ("s", "SERVER_TCP_FLAGS")]:
        v = d[col].fillna(0).astype("int64")
        for f, bit in FLAGS.items():
            out[f"{side}_{f}"] = ((v & bit) > 0).astype("float64")
    out["min_ttl"], out["max_ttl"] = d.MIN_TTL, d.MAX_TTL
    out["log_win_in"], out["log_win_out"] = np.log1p(d.TCP_WIN_MAX_IN.clip(lower=0)), np.log1p(d.TCP_WIN_MAX_OUT.clip(lower=0))
    return pd.DataFrame(out, index=d.index).astype("float32")


ABS_COUNTS = ["log_" + c for c in COUNT]
BINARY = ["has_reply", "is_tcp", "is_udp", "is_icmp"] + [f"{s}_{f}" for s in "cs" for f in FLAGS]
WIDE_ONLY = [f"{s}_{f}" for s in "cs" for f in FLAGS] + ["min_ttl", "max_ttl", "log_win_in", "log_win_out"]

if MODE == "prep":
    summary = []
    for name, file in FILES.items():
        path = DATA / file
        pf = pq.ParquetFile(path)
        n = pf.metadata.num_rows
        names = {c.strip(): c for c in pf.schema_arrow.names}
        missing = [c for c in RAW + ["Label", "Attack", "FLOW_START_MILLISECONDS"] if c not in names]
        if missing:
            sys.exit(f"{name} missing columns: {missing}")
        if n != OFFICIAL_ROWS[name]:
            sys.exit(f"{name} has {n} rows, official {OFFICIAL_ROWS[name]}")
        labels = pq.read_table(path, columns=[names["Attack"], names["Label"]])
        counts = labels.group_by([names["Attack"], names["Label"]]).aggregate([([], "count_all")]).to_pandas()
        counts.insert(0, "dataset", name)
        counts.to_csv(OUT / f"attack_counts_{name}.csv", index=False)
        attack_rows = int(pc_arrow.sum(labels[names["Label"]]).as_py())
        del labels
        log(name, "rows", n, "attack rows", attack_rows, f"{attack_rows / n:.2%}")
        print(counts.sort_values("count_all", ascending=False).to_string(index=False), flush=True)
        rng, keep, parts = np.random.default_rng(SEED), min(1.0, KEEP / n), []
        columns = [names[c] for c in RAW + ["Label", "Attack", "FLOW_START_MILLISECONDS"]]
        for batch in pf.iter_batches(batch_size=500_000, columns=columns):
            d = batch.filter(pa.array(rng.random(batch.num_rows) < keep)).to_pandas()
            d.columns = [c.strip() for c in d.columns]
            for c in RAW:
                d[c] = d[c].astype("float64")
            parts.append(d)
        d = pd.concat(parts, ignore_index=True)
        before = len(d)
        d = d.drop_duplicates(subset=RAW + ["Label"], ignore_index=True)
        X = build(d)
        hour = (d["FLOW_START_MILLISECONDS"].astype("float64") // 3_600_000).astype("int64")
        order = {h: i for i, h in enumerate(sorted(hour.unique()))}
        is_test = hour.map(order).values % 2 == 0
        X["label"] = d["Label"].astype("int8").values
        X["attack"] = d["Attack"].astype(str).str.strip().values
        X["is_test"] = is_test
        X.to_parquet(CACHE / f"{name}.parquet")
        cal, te = X[~is_test], X[is_test]
        summary.append({"dataset": name, "file_rows": n, "file_attack_rate": attack_rows / n, "sampled": before, "after_duplicates": len(X), "hours": len(order),
                        "calibration_rows": len(cal), "calibration_attack_rate": float(cal.label.mean()), "test_rows": len(te), "test_attack_rate": float(te.label.mean())})
        log(summary[-1])
        fam = X.groupby(["is_test", "attack"]).size().unstack(0).fillna(0).astype(int)
        print(fam.to_string(), flush=True)
    pd.DataFrame(summary).to_csv(OUT / "data_summary.csv", index=False)
    log("done")

if MODE == "run":
    # Attack names map to shared groups by their published meaning, fixed before any score was computed.
    GROUPS = {
        "UNSW": {"DoS": "dos", "Reconnaissance": "recon", "Backdoor": "backdoor"},
        "ToN": {"dos": "dos", "ddos": "dos", "scanning": "recon", "Backdoor": "backdoor", "password": "brute", "xss": "web", "injection": "web"},
        "CIC2018": {"DDOS_attack-HOIC": "dos", "DDOS_attack-LOIC-UDP": "dos", "DDoS_attacks-LOIC-HTTP": "dos", "DoS_attacks-GoldenEye": "dos", "DoS_attacks-Hulk": "dos", "DoS_attacks-SlowHTTPTest": "dos", "DoS_attacks-Slowloris": "dos",
                    "FTP-BruteForce": "brute", "SSH-Bruteforce": "brute", "Brute_Force_-Web": "web", "Brute_Force_-XSS": "web", "SQL_Injection": "web"},
    }
    data = {k: pd.read_parquet(CACHE / f"{k}.parquet") for k in FILES}
    features = [c for c in next(iter(data.values())).columns if c not in ("label", "attack", "is_test")]
    # UNSW records byte rates of zero-duration flows as infinity; the other two files never do.
    for d in data.values():
        d[features] = d[features].replace([np.inf, -np.inf], np.nan)
    behaviour = [c for c in features if c not in WIDE_ONLY]
    VIEWS = {"behaviour": behaviour, "no absolute counts": [c for c in behaviour if c not in ABS_COUNTS], "wide (flags, TTL, window)": features}
    ALIGN = ["none", "quantile", "coral"]
    rng = np.random.default_rng(SEED)

    def cap(idx, n):
        return idx if len(idx) <= n else np.sort(rng.choice(idx, n, replace=False))

    def otsu(p):
        h, e = np.histogram(p, bins=200)
        h = h / h.sum(); c = (e[:-1] + e[1:]) / 2; w = np.cumsum(h); mu = np.cumsum(h * c)
        return float(c[int(np.argmax((mu[-1] * w - mu) ** 2 / np.maximum(w * (1 - w), 1e-12)))])

    def quantile(X, pool, cols):
        X = X[cols].copy()
        for k in cols:
            if k in BINARY:
                continue
            ref = np.sort(pool[k].dropna().values[:: max(1, len(pool) // 200000)])
            X[k] = (np.searchsorted(ref, X[k].values, side="right") / len(ref)).astype("float32")
        return X

    def coral_map(Zs, Zp):
        def p(C, power):
            v, Q = np.linalg.eigh(C); v = np.clip(v, 1e-6, None)
            return (Q * v ** power) @ Q.T
        C_s, C_p = np.cov(Zs, rowvar=False) + 1e-3 * np.eye(Zs.shape[1]), np.cov(Zp, rowvar=False) + 1e-3 * np.eye(Zs.shape[1])
        ms, mt, A = Zs.mean(0), Zp.mean(0), p(C_s, -0.5) @ p(C_p, 0.5)
        return lambda Z: (Z - ms) @ A + mt

    def lgbm():
        return LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31, min_child_samples=100, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, n_jobs=10, random_state=SEED, verbose=-1, class_weight="balanced")

    def fit_predict(align, S, ys, source_pool, target_pool, parts, cols):
        if align == "quantile":
            m = lgbm().fit(quantile(S, source_pool, cols), ys)
            return [m.predict_proba(quantile(X, target_pool, cols))[:, 1] for X in parts]
        if align == "none":
            m = lgbm().fit(S[cols], ys)
            return [m.predict_proba(X[cols])[:, 1] for X in parts]
        imp = SimpleImputer(strategy="median").fit(S[cols])
        sc = StandardScaler().fit(imp.transform(S[cols]))
        z = lambda X: sc.transform(imp.transform(X[cols]))
        Zs = z(S)
        g = coral_map(Zs, z(target_pool))
        m = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, max_features="sqrt", n_jobs=10, random_state=SEED, class_weight="balanced").fit(g(Zs), ys)
        return [m.predict_proba(z(X))[:, 1] for X in parts]

    def score(y, s, t):
        h = s >= t
        qs = np.unique(np.quantile(s, np.linspace(0.01, 0.99, 99)))
        return {"roc_auc": roc_auc_score(y, s), "pr_auc": average_precision_score(y, s), "f1": f1_score(y, h, zero_division=0), "recall": float(h[y == 1].mean()), "fpr": float(h[y == 0].mean()),
                "flag_all_f1": 2 * y.mean() / (1 + y.mean()), "best_possible_f1": max(f1_score(y, s >= q, zero_division=0) for q in qs), "attack_rate": float(y.mean()), "n": len(y)}

    def shared_mask(df, groups, other):
        keep = set(groups.values()) & set(other.values())
        names = df["attack"].map(groups)
        return ((df["label"].values == 0) | names.isin(keep).values), keep

    rows, fams = [], []
    for scope in ("all", "shared"):
        for src in FILES:
            for tgt in FILES:
                if src == tgt:
                    continue
                S, T = data[src], data[tgt]
                if scope == "shared":
                    sm, keep = shared_mask(S, GROUPS[src], GROUPS[tgt])
                    tm, _ = shared_mask(T, GROUPS[tgt], GROUPS[src])
                else:
                    sm, tm, keep = np.ones(len(S), bool), np.ones(len(T), bool), {"all"}
                if not keep:
                    continue
                train = cap(np.where(sm)[0], 300_000)
                cal = cap(np.where(~T.is_test.values)[0], 300_000)
                test = cap(np.where(T.is_test.values & tm)[0], 1_000_000)
                pool = T.iloc[cal]
                y = T.label.values[test]
                names = T.attack.values[test]
                for align in ALIGN:
                    for view, cols in VIEWS.items():
                        tr = train if align != "coral" else cap(train, 150_000)
                        pc, pt = fit_predict(align, S.iloc[tr], S.label.values[tr], S, pool, [pool, T.iloc[test]], cols)
                        t = otsu(pc)
                        r = {"scope": scope, "shared_groups": "+".join(sorted(keep)), "source": src, "target": tgt, "align": align, "view": view, **score(y, pt, t)}
                        rows.append(r)
                        ben = pt[y == 0]
                        for f in sorted(set(names[y == 1])):
                            part = pt[names == f]
                            fams.append({"scope": scope, "source": src, "target": tgt, "align": align, "view": view, "family": f, "n": len(part), "recall": float((part >= t).mean()),
                                         "roc_vs_benign": roc_auc_score(np.r_[np.ones(len(part)), np.zeros(len(ben))], np.r_[part, ben])})
                        log(scope, src, "->", tgt, align, view, "roc", round(r["roc_auc"], 3), "f1", round(r["f1"], 3), "fpr", round(r["fpr"], 3), "flag-all", round(r["flag_all_f1"], 3))
                        pd.DataFrame(rows).to_csv(OUT / "all_configs.csv", index=False)
                        pd.DataFrame(fams).to_csv(OUT / "all_configs_by_family.csv", index=False)
    grid = pd.DataFrame(rows)
    locked = []
    for scope in ("all", "shared"):
        for tgt in FILES:
            sub = grid[(grid.scope == scope) & (grid.source != tgt) & (grid.target != tgt)]
            if not len(sub) or sub.groupby(["source", "target"]).ngroups < 2:
                continue
            mean = sub.groupby(["align", "view"])["roc_auc"].mean().sort_values(ascending=False)
            align, view = mean.index[0]
            pick = grid[(grid.scope == scope) & (grid.target == tgt) & (grid["align"] == align) & (grid.view == view)]
            for _, r in pick.iterrows():
                locked.append({**r.to_dict(), "selection_mean_roc_other_datasets": float(mean.iloc[0])})
    pd.DataFrame(locked).to_csv(OUT / "locked_results.csv", index=False)
    within = []
    for name, d in data.items():
        tr, te = cap(np.where(~d.is_test.values)[0], 300_000), cap(np.where(d.is_test.values)[0], 1_000_000)
        cols = VIEWS["behaviour"]
        m = lgbm().fit(d.iloc[tr][cols], d.label.values[tr])
        pte = m.predict_proba(d.iloc[te][cols])[:, 1]
        y = d.label.values[te]
        within.append({"dataset": name, **score(y, pte, 0.5)})
        log("within", within[-1])
    pd.DataFrame(within).to_csv(OUT / "within_dataset_reference.csv", index=False)
    log("done")
