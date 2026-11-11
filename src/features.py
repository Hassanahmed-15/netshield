"""One shared flow representation for UNSW-NB15 and CICIDS2017.

Every feature is computed by the same formula on both datasets from base
counters. No labels are used here.
"""
import numpy as np
import pandas as pd

SERVICE_BY_PORT = {
    80: "http", 8080: "http", 443: "ssl", 53: "dns", 21: "ftp", 20: "ftp-data",
    25: "smtp", 110: "pop3", 22: "ssh", 161: "snmp", 67: "dhcp", 68: "dhcp",
    194: "irc", 6667: "irc", 1812: "radius", 1813: "radius",
}
SERVICES = ["http", "ssl", "dns", "ftp", "ftp-data", "smtp", "pop3", "ssh", "snmp", "dhcp", "irc", "radius"]

GROUPS = {
    "size": ["log_dur", "log_fwd_pkts", "log_bwd_pkts", "log_fwd_bytes", "log_bwd_bytes",
             "log_total_pkts", "log_total_bytes"],
    "rate": ["log_pkt_rate", "log_byte_rate", "log_fwd_pkt_rate", "log_bwd_pkt_rate"],
    "shape": ["fwd_pkt_frac", "fwd_byte_frac", "log_fwd_pkt_size", "log_bwd_pkt_size",
              "log_avg_pkt_size", "no_response", "single_pkt"],
    "proto": ["is_tcp", "is_udp", "ended_fin", "ended_rst"],
    "service": ["svc_" + s for s in SERVICES] + ["svc_none"],
    "timing": ["log_fwd_iat", "log_bwd_iat", "log_fwd_jit", "log_bwd_jit"],
}
NOT_SCALED = {"no_response", "single_pkt", "is_tcp", "is_udp", "ended_fin", "ended_rst", "fwd_pkt_frac", "fwd_byte_frac"}


def _assemble(dur, fp, bp, fb, bb, is_tcp, is_udp, fin, rst, service, fiat, biat, fjit, bjit):
    dur = dur.clip(lower=0).fillna(0)
    safe = dur.clip(lower=1e-6)
    x = pd.DataFrame(index=dur.index)
    x["log_dur"] = np.log1p(dur * 1e6)
    x["log_fwd_pkts"], x["log_bwd_pkts"] = np.log1p(fp), np.log1p(bp)
    x["log_fwd_bytes"], x["log_bwd_bytes"] = np.log1p(fb), np.log1p(bb)
    tp, tb = fp + bp, fb + bb
    x["log_total_pkts"], x["log_total_bytes"] = np.log1p(tp), np.log1p(tb)
    x["log_pkt_rate"], x["log_byte_rate"] = np.log1p(tp / safe), np.log1p(tb / safe)
    x["log_fwd_pkt_rate"], x["log_bwd_pkt_rate"] = np.log1p(fp / safe), np.log1p(bp / safe)
    x["fwd_pkt_frac"] = fp / tp.replace(0, np.nan)
    x["fwd_byte_frac"] = fb / tb.replace(0, np.nan)
    x["log_fwd_pkt_size"] = np.log1p(fb / fp.replace(0, np.nan))
    x["log_bwd_pkt_size"] = np.log1p(bb / bp.replace(0, np.nan))
    x["log_avg_pkt_size"] = np.log1p(tb / tp.replace(0, np.nan))
    x["no_response"], x["single_pkt"] = (bp == 0).astype("int8"), (tp <= 1).astype("int8")
    x["is_tcp"], x["is_udp"], x["ended_fin"], x["ended_rst"] = is_tcp, is_udp, fin, rst
    for s in SERVICES:
        x["svc_" + s] = (service == s).astype("int8")
    x["svc_none"] = (~service.isin(SERVICES)).astype("int8")
    x["log_fwd_iat"], x["log_bwd_iat"] = np.log1p(fiat.clip(lower=0)), np.log1p(biat.clip(lower=0))
    x["log_fwd_jit"], x["log_bwd_jit"] = np.log1p(fjit.clip(lower=0)), np.log1p(bjit.clip(lower=0))
    return x.astype("float32")


def from_unsw(df):
    iat_f = df["sinpkt"] if "sinpkt" in df else df["sintpkt"]
    iat_b = df["dinpkt"] if "dinpkt" in df else df["dintpkt"]
    return _assemble(
        df["dur"].astype("float64"), df["spkts"].astype("float64"), df["dpkts"].astype("float64"),
        df["sbytes"].astype("float64"), df["dbytes"].astype("float64"),
        (df["proto"] == "tcp").astype("int8"), (df["proto"] == "udp").astype("int8"),
        (df["state"] == "FIN").astype("int8"), (df["state"] == "RST").astype("int8"),
        df["service"].astype(str).str.strip(),
        iat_f.astype("float64"), iat_b.astype("float64"),
        df["sjit"].astype("float64"), df["djit"].astype("float64"),
    )


def from_cicids(df, correct_bytes=True):
    fp, bp = df["Total Fwd Packets"].astype("float64"), df["Total Backward Packets"].astype("float64")
    fb, bb = df["Total Length of Fwd Packets"].astype("float64"), df["Total Length of Bwd Packets"].astype("float64")
    if correct_bytes:
        # CICFlowMeter reports payload bytes; UNSW counts IP + transport headers too.
        fb = fb + df["Fwd Header Length"].astype("float64").clip(lower=0) + 20 * fp
        bb = bb + df["Bwd Header Length"].astype("float64").clip(lower=0) + 20 * bp
    flags = df[["FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count", "ACK Flag Count"]].fillna(0).sum(axis=1) > 0
    win = (df["Init_Win_bytes_forward"].fillna(-1) >= 0) | (df["Init_Win_bytes_backward"].fillna(-1) >= 0)
    is_tcp = flags | win
    service = df["Destination Port"].fillna(-1).astype(int).map(SERVICE_BY_PORT).fillna("-")
    return _assemble(
        df["Flow Duration"].astype("float64") / 1e6, fp, bp, fb, bb,
        is_tcp.astype("int8"), (~is_tcp).astype("int8"),
        (df["FIN Flag Count"].fillna(0) > 0).astype("int8"), (df["RST Flag Count"].fillna(0) > 0).astype("int8"),
        service,
        df["Fwd IAT Mean"].astype("float64") / 1e3, df["Bwd IAT Mean"].astype("float64") / 1e3,
        df["Fwd IAT Std"].astype("float64") / 1e3, df["Bwd IAT Std"].astype("float64") / 1e3,
    )


def columns(groups):
    return [c for g in groups for c in GROUPS[g]]
