# -*- coding: utf-8 -*-
r"""Emit results/tables/tab_ledger.tex: every prediction of the seven
pre-registered protocols (code/PREREG_p3_*.md), its kind, whether the protocol
named it primary, how much it risked, and its outcome.

Outcomes are read from the stored results, never typed in:
    p3_prereg.json (new archives), p3_decline.json (Backblaze, two quarters),
    p3_rul.json (cells), p3_blind.json, p3_step.json, p3_filter.json,
    p3_fatigue.json.

Risk classes, fixed by a rule rather than by judgement after the fact:
    replication   -- the same asset class already had that verdict in the paper
                     before the protocol was frozen (C-MAPSS, NASA cells,
                     Backblaze Q4 2023);
    new class     -- a verdict on an asset class the paper had not screened;
    detector      -- a statement about the detector or the band, not a verdict;
    non-inferior  -- "the path model does not beat the baseline", which also
                     holds when the model has learned nothing;
    substantive   -- a predicted difference in accuracy or coverage.
Predictions whose condition was not met (B3, B4, F1-F3) are listed as not
testable and are not counted.

    python code/table_ledger.py
"""
from __future__ import annotations

import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BE = os.path.join(HERE, "..", "results", "bearings")
OUT = os.path.join(HERE, "..", "results", "tables", "tab_ledger.tex")


def J(name):
    return json.load(io.open(os.path.join(BE, name), encoding="utf-8"))


NA, DEC, RUL = J("p3_prereg.json"), J("p3_decline.json"), J("p3_rul.json")
BL, ST, FI, FA = J("p3_blind.json"), J("p3_step.json"), J("p3_filter.json"), J("p3_fatigue.json")


def held(d):
    return bool(d["holds"] if isinstance(d, dict) else d)


Q = DEC["cohorts"]
#  (protocol, id, statement, kind, primary, risk, outcome or None if untestable)
ROWS = [
    ("1 new archives", "P1", "XJTU cells not declined", "verdict", False, "replication", held(NA["P1"])),
    ("1 new archives", "P2", "N-CMAPSS not declined", "verdict", False, "replication", held(NA["P2"])),
    ("1 new archives", "P3", "neither median in 70--90\\,\\%", "detector", False, "detector", held(NA["P3"])),
    ("1 new archives", "P4", "onset at or after the labelled transition", "detector", False, "detector", held(NA["P4"])),
]
for q in ("Q1_2024", "Q2_2024"):
    lab = q.replace("_", " ")
    ROWS += [
        ("2 drives, " + lab, "P1", "declined", "verdict", True, "replication", Q[q]["P1"]),
        ("2 drives, " + lab, "P2", "onset in a quarter or more", "verdict", False, "replication", Q[q]["P2"]),
        ("2 drives, " + lab, "P3", "median onset at 90\\,\\% or later", "verdict", False, "replication", Q[q]["P3"]),
        ("2 drives, " + lab, "P4", "under 10\\,\\% of onsets before 70\\,\\%", "verdict", False, "replication", Q[q]["P4"]),
    ]
ROWS += [
    ("3 cells", "R1", "declined cohort errs more", "accuracy", True, "substantive", held(RUL["R1"])),
    ("3 cells", "R2", "declined cohort covers less", "accuracy", False, "substantive", held(RUL["R2"])),
    ("3 cells", "R3", "path model does not beat life table", "accuracy", False, "non-inferior", held(RUL["R3"])),
    ("4 blind N-CMAPSS", "B1", "no cohort declined", "verdict", False, "replication", held(BL["B1"])),
    ("4 blind N-CMAPSS", "B2", "error rises with onset", "accuracy", True, "substantive", held(BL["B2"])),
    ("4 blind N-CMAPSS", "B3", "late units err more", "accuracy", False, "substantive", None),
    ("4 blind N-CMAPSS", "B4", "declined cohorts err more", "accuracy", False, "substantive", None),
    ("5 fresh C-MAPSS", "D1", "late units err more", "accuracy", True, "substantive", held(ST["D1"])),
    ("5 fresh C-MAPSS", "D2", "error correlates with onset", "accuracy", False, "substantive", held(ST["D2"])),
    ("5 fresh C-MAPSS", "D3", "late units cover less", "accuracy", False, "substantive", held(ST["D3"])),
    ("5 fresh C-MAPSS", "D4", "path model does not beat life table", "accuracy", False, "non-inferior", held(ST["D4"])),
    ("6 filters", "V1", "no cohort declined", "verdict", False, "new class", held(FI["V1"])),
    ("6 filters", "F1--F3", "declined or late runs predicted worse", "accuracy", True, "substantive", None),
    ("7 titanium", "W1", "a cohort declined, none passed", "verdict", True, "new class", held(FA["W1"])),
    ("7 titanium", "W2", "onset in a quarter or more", "verdict", False, "new class", held(FA["W2"])),
    ("7 titanium", "W3", "path model does not beat baseline", "accuracy", False, "non-inferior", held(FA["W3"])),
    ("7 titanium", "W4", "late specimens cover less", "accuracy", False, "substantive", held(FA["W4"])),
]


def tally(pred):
    r = [x for x in ROWS if x[6] is not None and pred(x)]
    return sum(1 for x in r if x[6]), len(r)


SUMMARY = {
    "verdict and detector": tally(lambda x: x[3] in ("verdict", "detector")),
    "accuracy": tally(lambda x: x[3] == "accuracy"),
    "replication": tally(lambda x: x[5] == "replication"),
    "new class": tally(lambda x: x[5] == "new class"),
    "detector": tally(lambda x: x[5] == "detector"),
    "substantive": tally(lambda x: x[5] == "substantive"),
    "non-inferior": tally(lambda x: x[5] == "non-inferior"),
    "primary": tally(lambda x: x[4]),
}
assert SUMMARY["verdict and detector"] == (15, 16), SUMMARY
assert SUMMARY["accuracy"] == (6, 10), SUMMARY


def out(v):
    return "not testable" if v is None else ("held" if v else r"\textbf{failed}")


body, last = [], None
for p, i, s, k, prim, risk, v in ROWS:
    if last is not None and p.split(" ")[0] != last:
        body.append(r"\addlinespace[2pt]")
    last = p.split(" ")[0]
    body.append(f"{p} & {i}{'$^\\ast$' if prim else ''} & {s} & {risk} & {out(v)}\\\\")

summ = "; ".join(f"{k} {a}/{n}" for k, (a, n) in SUMMARY.items()
                 if k not in ("verdict and detector", "accuracy"))
tex = r"""%  generated by code/table_ledger.py -- do not edit
\begin{table}[htbp]
\centering
\caption{Every prediction of the seven pre-registered protocols, with its risk class and outcome.}
\label{tab:ledger}
\begin{adjustbox}{max width=\linewidth}
\begin{tabular}{@{}lllll@{}}
\toprule
protocol & & prediction & risk & outcome\\
\midrule
""" + "\n".join(body) + r"""
\bottomrule
\end{tabular}
\end{adjustbox}

\vspace{4pt}
\begin{minipage}{0.95\linewidth}\footnotesize
$^\ast$, primary as named in the protocol (for the drives, the test of a
decline). Risk: replication, an asset class whose verdict the paper had
already reported; new class, a verdict on an unscreened asset class;
detector, a claim about the detector or the band; non-inferior, that the path
model does not beat a baseline; substantive, a predicted difference in
accuracy or coverage. Held of testable: """ + summ + r""".
\end{minipage}
\end{table}
"""
io.open(OUT, "w", encoding="utf-8", newline="").write(tex)
print(f"  wrote {os.path.relpath(OUT, os.path.join(HERE, '..'))}")
print("  " + json.dumps(SUMMARY))
