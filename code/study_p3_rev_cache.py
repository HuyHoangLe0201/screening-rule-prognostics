# -*- coding: utf-8 -*-
"""Indicator series of every Table 8 archive, rebuilt from local data with the
study scripts' own code, for the detector revision (study_p3_rev_detector.py).

Nothing here re-implements an indicator.  Each archive's own script is reused:

  PRONOSTIA, XJTU-SY    datasets.load_dataset (the paper's log RMS)
  C-MAPSS               study_thirdset.load / indicator (via study_p3_review's
                        namespace loader, which stops before the study loop)
  NASA cells            study_battery.capacity (source executed up to its loop)
  XJTU Li-ion           study_p3_prereg.battery_capacity, hi = cap[0] - cap
  N-CMAPSS              study_p3_prereg.run_ncmapss with its onset() wrapped by
                        a recorder and its part file sent to the cache folder
  Backblaze             study_hdd executed in two pieces: the reading loop, then
                        the indicator loop with onset() wrapped by a recorder
                        and OUT pointed at the cache folder

Each archive's recorded onsets are checked against the stored results, so the
cached series are the ones the paper's verdicts were computed from.

    python code/study_p3_rev_cache.py core
    python code/study_p3_rev_cache.py hdd Q4_2023|Q1_2024|Q2_2024
    python code/study_p3_rev_cache.py ncmapss
Cache: $P3_REV_CACHE (default <tmp>/p3_rev_cache), one pickle per archive.
"""
from __future__ import annotations

import io
import json
import os
import pickle
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
BE = os.path.join(ROOT, "results", "bearings")
DATA = os.environ.get("P3_DATA", os.path.join(ROOT, "data"))
CACHE = os.environ.get("P3_REV_CACHE", os.path.join(tempfile.gettempdir(), "p3_rev_cache"))
os.makedirs(CACHE, exist_ok=True)


def cache_path(name):
    return os.path.join(CACHE, name.replace(" ", "_").replace("+", "p") + ".pkl")


def save(name, recs):
    """recs: list of dict(id, hi [array], frac_of [array or None], stored)"""
    pickle.dump(recs, open(cache_path(name), "wb"))
    print(f"  cached {name}: {len(recs)} records", flush=True)


def load(name):
    return pickle.load(open(cache_path(name), "rb"))


def _exec_head(path, marker="\nrows = []"):
    src = io.open(path, encoding="utf-8").read()
    i = src.index(marker)
    ns = {"__name__": "p3rev_head", "__file__": path}
    exec(compile(src[:i], path, "exec"), ns)
    return ns, src[i:]


def core():
    import datasets as ds
    import study_p3_review as R
    ref = R._onset_records()
    for name, label in (("pronostia", "PRONOSTIA"), ("xjtu", "XJTU-SY")):
        recs = [{"id": b, "hi": np.asarray(r["hi"], float), "frac_of": None}
                for b, r in sorted(ds.load_dataset(name).items())]
        for r, s in zip(recs, ref[label]):
            r["stored"] = s
        save(label, recs)
    ns = R._thirdset_namespace()
    for fd, by in (("FD001", False), ("FD003", False), ("FD002", True), ("FD004", True)):
        key = "C-MAPSS " + fd + ("+regime" if by else "")
        recs = [{"id": f"{fd}:{u}", "hi": ns["indicator"](arr), "frac_of": None}
                for u, arr in sorted(ns["load"](fd, by_regime=by).items())]
        for r, s in zip(recs, ref[key]):
            r["stored"] = s
        save(key, recs)
    #  NASA cells: study_battery up to its loop gives capacity()
    bns, _ = _exec_head(os.path.join(HERE, "study_battery.py"))
    bat = json.load(open(os.path.join(BE, "battery.json"), encoding="utf-8"))["cells"]
    recs = []
    for c in bat:
        cap = bns["capacity"](c["cell"])
        recs.append({"id": c["cell"], "hi": cap[0] - cap, "frac_of": None,
                     "stored": c["onset_pct"] / 100 if c["onset_idx"] else None})
    save("NASA cells", recs)
    #  XJTU Li-ion: the protocol's capacity series
    import glob
    import study_p3_prereg as P
    pre = {(r["batch"], r["cell"]): r for r in json.load(
        open(os.path.join(BE, "p3_prereg.json"), encoding="utf-8"))["battery"]["records"]}
    recs = []
    for batch in sorted(os.listdir(P.BAT)):
        if not os.path.isdir(os.path.join(P.BAT, batch)):
            continue
        for f in sorted(glob.glob(os.path.join(P.BAT, batch, "*.mat"))):
            cell = os.path.basename(f)[:-4]
            cap = P.battery_capacity(f, batch)
            recs.append({"id": f"{batch}/{cell}", "hi": cap[0] - cap, "frac_of": None,
                         "stored": pre[(batch, cell)]["onset_frac"]})
    save("XJTU Li-ion", recs)


def ncmapss():
    import study_p3_prereg as P
    got = []
    onset0 = P.NS["onset"]

    def rec(hi):
        got.append(np.asarray(hi, float).copy())
        return onset0(hi)
    P.NS = dict(P.NS, onset=rec)
    P.BE = CACHE                      # its part file goes to the cache, not results
    P.run_ncmapss()
    rows = json.load(open(os.path.join(CACHE, "p3_prereg_part_ncmapss.json")))
    stored = {(r["file"], r["split"], r["unit"]): r["onset_frac"] for r in json.load(
        open(os.path.join(BE, "p3_prereg.json"), encoding="utf-8"))["ncmapss"]["records"]}
    recs = []
    for r, hi in zip(rows, got):
        recs.append({"id": f"{r['file']}/{r['split']}/{r['unit']}", "hi": hi, "frac_of": None,
                     "stored": stored[(r["file"], r["split"], r["unit"])],
                     "label_frac": r["label_frac"]})
    save("N-CMAPSS", recs)


def hdd(q):
    path = os.path.join(HERE, "study_hdd.py")
    z = {"Q4_2023": os.path.join(DATA, "bb_Q4_2023.zip"),
         "Q1_2024": os.path.join(DATA, "backblaze_Q1_2024", "bb_Q1_2024.zip"),
         "Q2_2024": os.path.join(DATA, "backblaze_Q2_2024", "bb_Q2_2024.zip")}[q]
    sys.argv = [path, z]
    ns, tail = _exec_head(path)
    got = []
    onset0 = ns["onset"]

    def rec(hi):
        got.append({"id": ns["sn"], "hi": np.asarray(hi, float).copy(),
                    "frac_of": np.asarray(ns["ages"], float) / float(ns["ages"][-1])})
        return onset0(hi)
    ns["onset"] = rec
    ns["OUT"] = os.path.join(CACHE, f"hdd_{q}")
    exec(compile(tail, path, "exec"), ns)
    stored_file = (os.path.join(BE, "hdd.json") if q == "Q4_2023"
                   else os.path.join(BE, f"p3_decline_part_{q}.json"))
    stored = {d["serial"]: d["onset_frac"] for d in
              json.load(open(stored_file, encoding="utf-8"))["drives"]}
    for g in got:
        g["stored"] = stored[g["id"]]
    save("Backblaze " + q, got)


if __name__ == "__main__":
    if sys.argv[1] == "core":
        core()
    elif sys.argv[1] == "ncmapss":
        ncmapss()
    else:
        hdd(sys.argv[2])
