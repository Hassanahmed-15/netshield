"""Build compact parquet caches from the raw UNSW-NB15 and CICIDS2017 CSVs."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
CACHE = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

CIC_COLS = [
    "Destination Port", "Flow Duration", "Total Fwd Packets", "Total Backward Packets",
    "Total Length of Fwd Packets", "Total Length of Bwd Packets", "Fwd Header Length",
    "Bwd Header Length", "Fwd IAT Mean", "Fwd IAT Std", "Bwd IAT Mean", "Bwd IAT Std",
    "FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count",
    "Init_Win_bytes_forward", "Init_Win_bytes_backward", "Flow Packets/s", "Fwd Packets/s",
    "Bwd Packets/s", "Average Packet Size", "Label",
]

def load_cicids():
    frames = []
    for p in sorted((RAW / "CICIDS2017").glob("*.csv")):
        df = pd.read_csv(p, encoding="latin1", low_memory=False, skipinitialspace=True)
        df.columns = [c.strip().replace("﻿", "") for c in df.columns]
        df = df[CIC_COLS].copy()
        for c in CIC_COLS[:-1]:
            df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).astype("float32")
        df["label_name"] = df.pop("Label").astype(str).str.strip()
        df["capture"] = p.name.split(".")[0]
        frames.append(df)
        print(p.name, len(df), flush=True)
    out = pd.concat(frames, ignore_index=True)
    out["label"] = (out["label_name"].str.upper() != "BENIGN").astype("int8")
    out.to_parquet(CACHE / "cicids.parquet")
    print("cicids", out.shape, out.label.mean())

def load_unsw_official():
    d = RAW / "UNSW-NB15" / "Training and Testing Sets"
    for name, f in [("train", "UNSW_NB15_training-set.csv"), ("test", "UNSW_NB15_testing-set.csv")]:
        df = pd.read_csv(d / f, low_memory=False)
        df.columns = [c.strip().replace("﻿", "") for c in df.columns]
        df.to_parquet(CACHE / f"unsw_{name}.parquet")
        print(name, df.shape, df.label.mean())

def load_unsw_raw():
    names = pd.read_csv(RAW / "UNSW-NB15" / "NUSW-NB15_features.csv", encoding="latin1")["Name"].str.strip().str.lower().tolist()
    frames = []
    for i in range(1, 5):
        df = pd.read_csv(RAW / "UNSW-NB15" / f"UNSW-NB15_{i}.csv", header=None, names=names, encoding="latin1", low_memory=False)
        df["part"] = i
        frames.append(df)
        print("raw part", i, df.shape, flush=True)
    out = pd.concat(frames, ignore_index=True)
    for c in ["sport", "dsport", "ct_ftp_cmd"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["attack_cat"] = out["attack_cat"].astype(str).str.strip().replace({"nan": "Normal", "": "Normal"})
    for c in out.select_dtypes("object").columns:
        out[c] = out[c].astype(str).str.strip()
    out.to_parquet(CACHE / "unsw_raw.parquet")
    print("raw", out.shape, out.label.mean())

if __name__ == "__main__":
    load_unsw_official()
    load_unsw_raw()
    load_cicids()
