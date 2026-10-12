# -*- coding: utf-8 -*-
r"""The pre-registered test of the screening rule on two archives it never saw.

Implements code/PREREG_p3_new_archives.md exactly, and refuses to run if that
file has changed since it was frozen (2026-10-12T18:20:45+07:00).

    python code/study_p3_prereg.py battery Batch-1     one battery batch
    python code/study_p3_prereg.py ncmapss             both N-CMAPSS files
    python code/study_p3_prereg.py summary             merge, test P1-P4
        ->  results/bearings/p3_prereg.json

The battery files carry per-second data and load slowly, so the batches are
meant to run as separate processes; each writes a part file that `summary`
folds in and removes.
"""
from __future__ import annotations

import glob
import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import study_p3_review as R                                      # noqa: E402

PREREG = os.path.join(HERE, "PREREG_p3_new_archives.md")
FROZEN_SHA256 = "c6116aeba39294d6cc3d6af0e62ceda65f00d065941a47a75936df24b2e2d0b1"
FROZEN_AT = "2026-10-12T18:20:45+07:00"
DATA = os.environ.get("P3_DATA", os.path.join(ROOT, "data"))
BAT = os.path.join(DATA, "NASABattery", "Battery Dataset")
NCM = [os.path.join(DATA, "ncmapss", f) for f in
       ("N-CMAPSS_DS01-005.h5", "N-CMAPSS_DS02-006.h5")]
BE = os.path.join(ROOT, "results", "bearings")
OUT = os.path.join(BE, "p3_prereg.json")
SEED, B_BOOT = 20261015, 10_000


def check_frozen():
    h = hashlib.sha256(open(PREREG, "rb").read()).hexdigest()
    if h != FROZEN_SHA256:
        sys.exit(f"the protocol has changed since it was frozen: {h}")
    return h


NS = R._thirdset_namespace()          # the paper's indicator() and onset()


# ══════════════════════════════════════════════════════════════════════════
# A. XJTU batteries
# ══════════════════════════════════════════════════════════════════════════
def battery_capacity(path, batch):
    """The capacity series of the protocol, for one cell."""
    import scipy.io as sio
    m = sio.loadmat(path, simplify_cells=True)
    desc = [c["description"] for c in m["data"]]
    cap = np.ravel(m["summary"]["discharge_capacity_Ah"]).astype(float)
    assert len(cap) == len(desc), path
    initial = ["0.2C discharge" in d and "[test capacity]" in d for d in desc]
    if batch in ("Batch-1", "Batch-2"):
        keep = ["[test capacity]" not in d for d in desc]
    elif batch == "Batch-3":
        keep = [d == "2C charge and 1C discharge" for d in desc]
    else:
        keep = ["[test capacity]" in d for d in desc]
    keep = [k and not i for k, i in zip(keep, initial)]
    return cap[np.array(keep, bool)]


def run_battery(batch):
    rows = []
    for f in sorted(glob.glob(os.path.join(BAT, batch, "*.mat"))):
        cell = os.path.basename(f)[:-4]
        try:
            cap = battery_capacity(f, batch)
        except Exception as e:                       # reported, never silent
            rows.append({"cell": cell, "batch": batch, "load_error": repr(e)})
            continue
        hi = cap[0] - cap
        i = NS["onset"](hi)
        rows.append({"cell": cell, "batch": batch, "n": int(len(hi)),
                     "onset_idx": int(i), "onset_frac": i / len(hi) if i > 0 else None})
        print(f"  {batch} {cell}: n={len(hi)} onset={'none' if i == 0 else f'{i / len(hi):.3f}'}",
              flush=True)
    json.dump(rows, open(os.path.join(BE, f"p3_prereg_part_{batch}.json"), "w"), indent=1)


# ══════════════════════════════════════════════════════════════════════════
# B. N-CMAPSS
# ══════════════════════════════════════════════════════════════════════════
def _quadratic(w):
    cols = [np.ones(len(w))] + [w[:, j] for j in range(4)]
    cols += [w[:, i] * w[:, j] for i in range(4) for j in range(i, 4)]
    return np.column_stack(cols)                     # 15 columns


def run_ncmapss():
    import h5py
    rows = []
    for path in NCM:
        tag = os.path.basename(path)[:-3]
        for split in ("dev", "test"):
            with h5py.File(path, "r") as f:
                A = f[f"A_{split}"][:]
                W = f[f"W_{split}"][:]
                X = f[f"X_s_{split}"][:]
            for u in np.unique(A[:, 0]).astype(int):
                sel = A[:, 0] == u
                cyc, hs = A[sel, 1].astype(int), A[sel, 3]
                w, x = W[sel], X[sel]
                cycles = np.unique(cyc)
                n = len(cycles)
                nh = min(40, max(5, n // 10))
                healthy = cyc <= cycles[nh - 1]
                mu, sd = w[healthy].mean(axis=0), w[healthy].std(axis=0) + 1e-12
                Phi = _quadratic((w - mu) / sd)
                beta, *_ = np.linalg.lstsq(Phi[healthy], x[healthy], rcond=None)
                res = x - Phi @ beta
                pos = np.searchsorted(cycles, cyc)
                cnt = np.bincount(pos, minlength=n)
                M = np.column_stack([np.bincount(pos, weights=res[:, j], minlength=n) / cnt
                                     for j in range(res.shape[1])])
                hi = NS["indicator"](M)
                i = NS["onset"](hi)
                hs_c = np.array([hs[pos == k].min() for k in range(n)])
                lab = int(np.argmax(hs_c == 0)) if (hs_c == 0).any() else None
                rows.append({"file": tag, "split": split, "unit": int(u), "n": n,
                             "onset_idx": int(i), "onset_frac": i / n if i > 0 else None,
                             "label_idx": lab, "label_frac": None if lab is None else lab / n})
                print(f"  {tag} {split} unit {u}: n={n} onset="
                      f"{'none' if i == 0 else f'{i / n:.3f}'} label="
                      f"{'none' if lab is None else f'{lab / n:.3f}'}", flush=True)
    json.dump(rows, open(os.path.join(BE, "p3_prereg_part_ncmapss.json"), "w"), indent=1)


# ══════════════════════════════════════════════════════════════════════════
# summary
# ══════════════════════════════════════════════════════════════════════════
def _summ(fracs, rng):
    x = np.array([np.inf if f is None else f for f in fracs], float)
    det = x[np.isfinite(x)]
    lo, hi = R.wilson(len(det), len(x))
    idx = rng.integers(0, len(x), size=(B_BOOT, len(x)))
    cm = np.median(x[idx], axis=1)
    clo, chi = np.quantile(cm, [0.025, 0.975], method="inverted_cdf")

    def pct(z):
        return None if not np.isfinite(z) else 100 * float(z)
    out = {"n": len(x), "n_detected": len(det), "rate": len(det) / len(x),
           "rate_ci": [lo, hi], "censored_median_pct": pct(np.median(x)),
           "censored_median_ci": [pct(clo), pct(chi)]}
    if len(det):
        d_idx = rng.integers(0, len(det), size=(B_BOOT, len(det)))
        out["median_pct"] = 100 * float(np.median(det))
        out["median_ci"] = [100 * float(v) for v in
                            np.percentile(np.median(det[d_idx], axis=1), [2.5, 97.5])]
    if out["rate"] < 0.25:
        out["verdict"] = "check indicator"
    else:
        cmv = np.median(x)
        out["verdict"] = "declines" if (not np.isfinite(cmv) or cmv >= 0.70) else "not declined"
    return out


def summary():
    h = check_frozen()
    rng = np.random.default_rng(SEED)
    bat = []
    for f in sorted(glob.glob(os.path.join(BE, "p3_prereg_part_Batch-*.json"))):
        bat += json.load(open(f))
    ncm = json.load(open(os.path.join(BE, "p3_prereg_part_ncmapss.json")))
    errors = [r for r in bat + ncm if "load_error" in r]
    bat_ok = [r for r in bat if "load_error" not in r]
    res = {"protocol_sha256": h, "protocol_frozen_at": FROZEN_AT,
           "battery": {"records": bat, **_summ([r["onset_frac"] for r in bat_ok], rng)},
           "ncmapss": {"records": ncm, **_summ([r["onset_frac"] for r in ncm], rng)},
           "load_errors": errors}
    for k in ("battery", "ncmapss"):
        res[k]["by_group"] = {}
    for b in sorted({r["batch"] for r in bat_ok}):
        fr = [r["onset_frac"] for r in bat_ok if r["batch"] == b]
        res["battery"]["by_group"][b] = {"n": len(fr), "n_detected": sum(f is not None for f in fr),
                                         "median_pct": (100 * float(np.median([f for f in fr if f is not None]))
                                                        if any(f is not None for f in fr) else None)}
    b, c = res["battery"], res["ncmapss"]

    def below70(s):
        return s["censored_median_pct"] is not None and s["censored_median_pct"] < 70
    pairs = [(r["onset_idx"], r["label_idx"], r["n"]) for r in ncm
             if r["onset_idx"] > 0 and r["label_idx"] is not None]
    after = [o >= l for o, l, _ in pairs]
    lags = [(o - l) / n for o, l, n in pairs]
    res["P1"] = {"holds": bool(b["rate"] >= 0.25 and below70(b))}
    res["P2"] = {"holds": bool(c["rate"] >= 0.25 and below70(c))}
    res["P3"] = {"holds": bool(all(s["censored_median_pct"] is None
                                   or not (70 <= s["censored_median_pct"] <= 90) for s in (b, c)))}
    res["P4"] = {"holds": bool(len(after) > 0 and np.mean(after) >= 0.75),
                 "n_units": len(after), "share_at_or_after": float(np.mean(after)) if after else None,
                 "median_lag_frac": float(np.median(lags)) if lags else None,
                 "lag_iqr": [float(v) for v in np.percentile(lags, [25, 75])] if lags else None}
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1)
    for f in glob.glob(os.path.join(BE, "p3_prereg_part_*.json")):
        os.remove(f)
    for k in ("battery", "ncmapss"):
        s = res[k]
        print(f"{k}: {s['n_detected']}/{s['n']} detected, rate {s['rate']:.2f} {s['rate_ci']}, "
              f"censored median {s['censored_median_pct']} {s['censored_median_ci']}, "
              f"detected median {s.get('median_pct')} {s.get('median_ci')} -> {s['verdict']}")
    print("by battery batch:", res["battery"]["by_group"])
    for p in ("P1", "P2", "P3", "P4"):
        print(p, res[p])
    print("load errors:", errors)


if __name__ == "__main__":
    check_frozen()
    if sys.argv[1] == "battery":
        run_battery(sys.argv[2])
    elif sys.argv[1] == "ncmapss":
        run_ncmapss()
    else:
        summary()
