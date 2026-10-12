"""LaTeX table for the held-back archives: C-MAPSS and NASA battery."""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CM = os.path.join(HERE, "..", "results", "bearings", "thirdset.json")
BAT = os.path.join(HERE, "..", "results", "bearings", "battery.json")

COND = {"FD001": "one", "FD002": "six", "FD003": "one", "FD004": "six",
        "FD002+regime": "six, repaired", "FD004+regime": "six, repaired"}


def build():
    d = json.load(io.open(CM, encoding="utf-8"))
    b = json.load(io.open(BAT, encoding="utf-8"))

    def verdict(n_det, n, med):
        #  the rule of the paper's Section 5: an onset in fewer than a quarter of
        #  the records sends the user back to the indicator; otherwise a median
        #  at 70 % of life or later declines the first-passage model.  A pass is
        #  "not declined", never "applies", which the study does not show.
        if n_det / n < 0.25:
            return "check indicator"
        return "declines" if med >= 70 else "not declined"

    rows = ""
    for fd in ("FD001", "FD003", "FD002", "FD004", "FD002+regime", "FD004+regime"):
        r = d[fd]
        m, p = r["median_onset_pct"], r["median_deg_phase_pct"]
        rows += (f"C-MAPSS {fd.replace('+regime', '')} & {COND[fd]} & "
                 f"{r['n_units']} & {r['n_detected']}/{r['n_units']} & "
                 f"{r['median_healthy_mad']:.2f} & {m:.0f} & {p:.0f} & "
                 + verdict(r["n_detected"], r["n_units"], m) + "\\\\\n")
    rows += (f"NASA battery & --- & {b['n_cells']} & "
             f"{b['n_detected']}/{b['n_cells']} & --- & "
             f"{b['median_onset_pct']:.0f} & {b['median_deg_phase_pct']:.0f} & "
             + verdict(b["n_detected"], b["n_cells"], b["median_onset_pct"]) + "\\\\\n")

    return f"""\\begin{{table}}[htbp]
\\centering\\small
\\caption{{The screening rule on the two bearing archives and, unchanged, on archives
that took no part in its construction.}}
\\label{{tab:thirdset}}
\\begin{{adjustbox}}{{max width=\\linewidth}}
\\begin{{tabular}}{{@{{}}l l r r r r r l@{{}}}}
\\toprule
archive & op.\\ cond. & units & onset found & healthy scale
        & median onset [\\%] & deg.\\ phase [\\%] & verdict\\\\
\\midrule
PRONOSTIA & one & 17 & 12/17 & --- & 97 & 3 & declines\\\\
XJTU-SY & one & 15 & 15/15 & --- & 41 & 59 & not declined\\\\
\\midrule
{rows}\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}

\\vspace{{4pt}}
\\begin{{minipage}}{{0.95\\linewidth}}\\footnotesize
The first two rows are the archives on which the rule was built, repeated for
comparison. The healthy-scale column is on each archive's own indicator scale and
is comparable only within C-MAPSS.
\\end{{minipage}}
\\end{{table}}
"""


if __name__ == "__main__":
    out = os.path.join(HERE, "..", "results", "tables", "tab_thirdset.tex")
    io.open(out, "w", encoding="utf-8").write(build())
    print("  -> results/tables/tab_thirdset.tex")
