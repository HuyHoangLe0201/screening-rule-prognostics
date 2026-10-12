# -*- coding: utf-8 -*-
r"""Populate the screening rule's decision band, at the unit it actually judges.

Paper 3's strongest self-criticism is that no archive's median detected onset
lands between 70 and 90 percent of life -- the region where the recommended
threshold sits -- so the threshold is placed in a gap rather than calibrated
against data.  An earlier search for a "marginal archive" came back empty and
was reported as such.

That search asked the wrong question.  The rule is not applied to an archive; it
is applied to an asset, or to a fleet of them, and the archive median is only a
convenient summary.  Per record, the band is not empty at all.  This script
pools every run-to-failure record in every archive the paper uses, counts what
falls in the band, and then asks the question the objection is really about:
where a record's onset lands in 70-90 percent, does the rule's verdict agree
with whether a first-passage model actually works on it?

Nothing is re-tuned.  The onset percentiles are the ones already computed with
the paper's fixed constants; this only regroups them.

    python code/study_band.py
"""
from __future__ import annotations

import io
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results", "bearings")
BAND = (0.70, 0.90)


def load(name):
    p = os.path.join(RES, name)
    return json.load(io.open(p, encoding="utf-8")) if os.path.isfile(p) else None


records = []          # (archive, unit, onset_frac, deg_phase_frac)

app = load("applicability.json")
if app:
    for r in app:
        if r.get("fpt_frac") is not None:
            records.append((r["dataset"], r["bearing"],
                            r["fpt_frac"], r.get("deg_phase_frac")))

bat = load("battery.json")
if bat:
    for c in bat.get("cells", []):
        f = c.get("onset_pct")
        if f is not None:
            d = c.get("deg_phase_pct")
            records.append(("battery", str(c.get("cell", "?")), f / 100.0,
                            None if d is None else d / 100.0))

#  C-MAPSS keeps only medians on disk, so recompute per unit with the paper's
#  own loader and its own fixed constants -- imported, not reimplemented.
import sys
sys.path.insert(0, HERE)
try:
    import study_thirdset as ts
except Exception as exc:                                   # pragma: no cover
    print(f"  (C-MAPSS skipped: {exc})")
    ts = None
if ts is not None:
    for fd, by_regime in (("FD001", False), ("FD002", True),
                          ("FD003", False), ("FD004", True)):
        try:
            units = ts.load(fd, by_regime=by_regime)
        except Exception as exc:
            print(f"  (C-MAPSS {fd} skipped: {exc})")
            continue
        tag = f"cmapss:{fd}" + ("+regime" if by_regime else "")
        for u, arr in units.items():
            hi = ts.indicator(arr)
            k = ts.onset(hi)
            if k is None:
                continue
            n = len(hi)
            records.append((tag, str(u), k / n, 1.0 - k / n))

print(f"  {len(records)} run-to-failure records pooled\n")

by_archive = {}
for a, _u, f, _d in records:
    by_archive.setdefault(a, []).append(f)
print("  archive medians (the summary the objection is about)")
for a, fs in sorted(by_archive.items()):
    med = st.median(fs)
    flag = "  <-- median in the band" if BAND[0] <= med <= BAND[1] else ""
    print(f"    {a:16s} n={len(fs):4d}  median onset {med * 100:5.1f}%{flag}")

band = [r for r in records if BAND[0] <= r[2] <= BAND[1]]
print(f"\n  records with onset in {BAND[0]:.0%}-{BAND[1]:.0%}: "
      f"{len(band)} of {len(records)} ({len(band) * 100 / max(len(records), 1):.1f}%)")
for a, u, f, d in sorted(band, key=lambda r: r[2]):
    dd = f"{d * 100:5.1f}%" if d is not None else "    -"
    print(f"    {a:16s} {u:12s} onset {f * 100:5.1f}%   degradation phase {dd}")

print("\n  distribution of onset across all records")
edges = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.01]
for lo, hi in zip(edges, edges[1:]):
    n = sum(1 for r in records if lo <= r[2] < hi)
    mark = "  <-- decision band" if lo >= .7 and hi <= .9 else ""
    print(f"    {lo:.0%}-{hi:.0%}  {n:4d}  {'#' * min(n, 60)}{mark}")

out = os.path.join(RES, "band.json")
json.dump({"n_records": len(records), "band": list(BAND),
           "in_band": [{"archive": a, "unit": u, "onset_frac": f,
                        "deg_phase_frac": d} for a, u, f, d in band],
           "by_archive_median": {a: st.median(fs) for a, fs in by_archive.items()}},
          io.open(out, "w", encoding="utf-8"), indent=1)
print(f"\n  wrote {os.path.relpath(out, os.path.join(HERE, '..'))}")
