# -*- coding: utf-8 -*-
r"""The pre-registered test of a decline: the screen on two quarters of
Backblaze drives it never saw.

Implements code/PREREG_p3_decline.md exactly, and refuses to run if that file,
or the analysis code it names (study_hdd.py), has changed since the protocol
was frozen.

    python code/study_p3_decline.py run Q1_2024     reads $P3_DATA/bb_Q1_2024.zip
    python code/study_p3_decline.py run Q2_2024
    python code/study_p3_decline.py summary         tests P1-P4 per cohort
        ->  results/bearings/p3_decline.json

`run` executes study_hdd.py's source unchanged, with only the name of its
output file replaced, so the selection of drives, the indicator and the onset
detector are those that produced the Q4 2023 verdict.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PREREG = os.path.join(HERE, "PREREG_p3_decline.md")
HDD = os.path.join(HERE, "study_hdd.py")
FROZEN_SHA256 = "0e6a6801c6667061fd8edb9486f25c71952fad297237264ab2126f3e88acf890"
FROZEN_AT = "2026-10-13T01:08:01+07:00"
HDD_SHA256 = "b4c4e520467882a241d5dbc7c16e232a13ba5b7e3c62d1836467c97c127af74f"
DATA = os.environ.get("P3_DATA", os.path.join(ROOT, "data"))
BE = os.path.join(ROOT, "results", "bearings")
OUT = os.path.join(BE, "p3_decline.json")
COHORTS = ("Q1_2024", "Q2_2024")
SEED, B_BOOT = 20261014, 10_000
THRESHOLD, MIN_USABLE = 0.70, 30


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def check_frozen():
    if FROZEN_SHA256 is None or sha256(PREREG) != FROZEN_SHA256:
        sys.exit(f"the protocol has changed since it was frozen: {sha256(PREREG)}")
    if sha256(HDD) != HDD_SHA256:
        sys.exit(f"study_hdd.py has changed since the protocol was frozen: {sha256(HDD)}")


def part(q):
    return os.path.join(BE, f"p3_decline_part_{q}.json")


def run(q):
    src = io.open(HDD, encoding="utf-8").read()
    old = 'os.path.join(OUT, "hdd.json")'
    if src.count(old) != 1:
        sys.exit("study_hdd.py no longer writes hdd.json in one place")
    src = src.replace(old, f'os.path.join(OUT, "{os.path.basename(part(q))}")')
    src = src.replace("wrote results/bearings/hdd.json",
                      f"wrote results/bearings/{os.path.basename(part(q))}")
    sys.argv = [HDD, os.path.join(DATA, f"bb_{q}.zip")]
    exec(compile(src, HDD, "exec"), {"__name__": "__main__", "__file__": HDD})


def verdict(lo, hi):
    """The three outcomes of Algorithm 3 on the censored interval (fractions)."""
    if lo >= THRESHOLD:
        return "declines"
    return "not declined" if hi < THRESHOLD else "undecided"


def summary():
    rng = np.random.default_rng(SEED)              # cohorts in the order of COHORTS
    res = {"protocol_sha256": FROZEN_SHA256, "protocol_frozen_at": FROZEN_AT,
           "study_hdd_sha256": HDD_SHA256, "cohorts": {}}
    for q in COHORTS:
        d = json.load(io.open(part(q), encoding="utf-8"))
        rows = d["drives"]
        x = np.array([np.inf if r["onset_frac"] is None else r["onset_frac"]
                      for r in rows], float)
        det = x[np.isfinite(x)]
        c = {"n_failed": d["n_failed"], "n_usable": len(x), "n_detected": int(det.size)}
        if len(x) < MIN_USABLE:
            c["tested"] = False
            res["cohorts"][q] = c
            continue
        med = float(np.median(x))
        meds = np.median(x[rng.integers(0, len(x), size=(B_BOOT, len(x)))], axis=1)
        lo, hi = (float(v) for v in np.quantile(meds, [0.025, 0.975], method="inverted_cdf"))

        def pct(z):
            return None if not np.isfinite(z) else 100 * z
        lead = [r["life_hours"] * (1 - r["onset_frac"]) for r in rows
                if r["onset_frac"] is not None]
        c.update({
            "tested": True,
            "rate": det.size / len(x),
            "censored_median_pct": pct(med), "censored_median_ci": [pct(lo), pct(hi)],
            "median_detected_pct": pct(float(np.median(det))) if det.size else None,
            "share_detected_before_70": float(np.mean(det < THRESHOLD)) if det.size else None,
            "lead_hours_median": float(np.median(lead)) if lead else None,
            "verdict": verdict(lo, hi)})
        c["P1"] = c["verdict"] == "declines"
        c["P2"] = c["rate"] >= 0.25
        c["P3"] = det.size > 0 and float(np.median(det)) >= 0.90
        c["P4"] = det.size > 0 and c["share_detected_before_70"] < 0.10
        #  descriptive only, for the table's other columns; not in the
        #  protocol, and drawn from a separate generator so that nothing above
        #  depends on it
        r2 = np.random.default_rng(SEED + 1)
        z = 1.959964
        p_, n_ = det.size / len(x), len(x)
        cc = (p_ + z * z / (2 * n_)) / (1 + z * z / n_)
        hw = z * np.sqrt(p_ * (1 - p_) / n_ + z * z / (4 * n_ * n_)) / (1 + z * z / n_)
        c["rate_ci"] = [max(0.0, float(cc - hw)), min(1.0, float(cc + hw))]
        if det.size:
            dm = np.median(det[r2.integers(0, det.size, size=(B_BOOT, det.size))], axis=1)
            c["median_detected_ci"] = [100 * float(v) for v in np.percentile(dm, [2.5, 97.5])]
        res["cohorts"][q] = c
        print(f"{q}: usable {len(x)}, detected {det.size} ({c['rate']:.2f}); censored median "
              f"{c['censored_median_pct']} {c['censored_median_ci']}; verdict {c['verdict']}; "
              f"P1 {c['P1']} P2 {c['P2']} P3 {c['P3']} P4 {c['P4']}")
    tested = [c for c in res["cohorts"].values() if c.get("tested")]
    res["decline_confirmed"] = bool(tested) and len(tested) == len(COHORTS) and all(
        c["P1"] for c in tested)
    json.dump(res, io.open(OUT, "w", encoding="utf-8"), indent=1)
    print(f"decline confirmed in both cohorts: {res['decline_confirmed']}  -> {OUT}")


if __name__ == "__main__":
    check_frozen()
    if sys.argv[1:2] == ["run"] and sys.argv[2:3] and sys.argv[2] in COHORTS:
        run(sys.argv[2])
    elif sys.argv[1:2] == ["summary"]:
        summary()
    else:
        sys.exit(__doc__)
