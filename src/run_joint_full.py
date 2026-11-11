"""Joint and within-dataset scores on the WHOLE held-out test sets.

run_final.py halved these test sets by row order, but the official UNSW test
file is sorted in blocks, so its halves are not comparable. Models, training
rows, seeds and thresholds here are the same as run_final.py.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent))
import features as F

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "cache"
OUT = ROOT / "outputs" / "final_model" / "tables"
SEED = 42
cols = F.columns(["size", "rate", "shape", "proto", "service", "timing"])


def metrics(y, p, t, **tags):
    h = p >= t
    return {**tags, "n": len(y), "attack_rate": float(y.mean()), "roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "f1": f1_score(y, h, zero_division=0), "precision": precision_score(y, h, zero_division=0), "recall": recall_score(y, h, zero_division=0),
            "fpr": float(h[y == 0].mean()), "threshold": float(t)}


def lgbm():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=127, min_child_samples=100, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, n_jobs=8, random_state=SEED, verbose=-1, class_weight="balanced")


def best_t(y, p, w=None):
    ts = np.linspace(0.02, 0.98, 97)
    return float(ts[int(np.argmax([f1_score(y, p >= t, sample_weight=w, zero_division=0) for t in ts]))])


cic = pd.read_parquet(CACHE / "cicids.parquet")
cic = cic.drop_duplicates([c for c in cic.columns if c not in ("capture", "label_name")]).reset_index(drop=True)
unsw = pd.concat([pd.read_parquet(CACHE / "unsw_train.parquet").assign(part="train"), pd.read_parquet(CACHE / "unsw_test.parquet").assign(part="test")], ignore_index=True)
Xc, yc, Xu, yu = F.from_cicids(cic, True), cic["label"].values, F.from_unsw(unsw), unsw["label"].values

key_c = pd.util.hash_pandas_object(Xc[F.columns(["size", "rate", "shape", "proto", "service", "timing"])].round(4), index=False).values
ci = np.sort(np.unique(key_c, return_index=True)[1])
ctr, cte = train_test_split(ci, test_size=0.3, stratify=yc[ci], random_state=SEED)
ctr, cva = train_test_split(ctr, test_size=0.15, stratify=yc[ctr], random_state=SEED)
utrain = np.where((unsw["part"] == "train").values)[0]
uall = np.where((unsw["part"] == "test").values)[0]
key_u = pd.util.hash_pandas_object(Xu[cols].round(4), index=False).values
utest = uall[~np.isin(key_u[uall], key_u[utrain])]
utest = utest[np.sort(np.unique(key_u[utest], return_index=True)[1])]
utrain = utrain[np.sort(np.unique(key_u[utrain], return_index=True)[1])]
utr, uva = train_test_split(utrain, test_size=0.15, stratify=yu[utrain], random_state=SEED)
fit = np.random.default_rng(SEED).choice(ctr, min(700_000, len(ctr)), replace=False)

w = np.r_[np.full(len(utr), 0.5 / len(utr)), np.full(len(fit), 0.5 / len(fit))] * (len(utr) + len(fit))
m = lgbm().fit(pd.concat([Xu.iloc[utr][cols], Xc.iloc[fit][cols]]), np.r_[yu[utr], yc[fit]], sample_weight=w)
pv = np.r_[m.predict_proba(Xu.iloc[uva][cols])[:, 1], m.predict_proba(Xc.iloc[cva][cols])[:, 1]]
t = best_t(np.r_[yu[uva], yc[cva]], pv, np.r_[np.full(len(uva), 0.5 / len(uva)), np.full(len(cva), 0.5 / len(cva))])
rows = [metrics(yu[utest], m.predict_proba(Xu.iloc[utest][cols])[:, 1], t, regime="joint", dataset="UNSW", test="official test, rows also in train removed"),
        metrics(yu[uall], m.predict_proba(Xu.iloc[uall][cols])[:, 1], t, regime="joint", dataset="UNSW", test="official test, all rows"),
        metrics(yc[cte], m.predict_proba(Xc.iloc[cte][cols])[:, 1], t, regime="joint", dataset="CICIDS", test="random 30%, duplicates removed")]
mu = lgbm().fit(Xu.iloc[utr][cols], yu[utr]); tu = best_t(yu[uva], mu.predict_proba(Xu.iloc[uva][cols])[:, 1])
rows += [metrics(yu[utest], mu.predict_proba(Xu.iloc[utest][cols])[:, 1], tu, regime="within", dataset="UNSW", test="official test, rows also in train removed"),
         metrics(yu[uall], mu.predict_proba(Xu.iloc[uall][cols])[:, 1], tu, regime="within", dataset="UNSW", test="official test, all rows")]
mc = lgbm().fit(Xc.iloc[fit][cols], yc[fit]); tc = best_t(yc[cva], mc.predict_proba(Xc.iloc[cva][cols])[:, 1])
rows += [metrics(yc[cte], mc.predict_proba(Xc.iloc[cte][cols])[:, 1], tc, regime="within", dataset="CICIDS", test="random 30%, duplicates removed")]
r = pd.DataFrame(rows)
r.to_csv(OUT / "joint_full_test.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20); pd.set_option("display.max_colwidth", 45)
print(r.round(3).to_string())
