"""LaTeX table for the onset-detector sensitivity sweep.

Kept out of make_tables.py for the same reason as the other table_*.py files:
the caption is long and the escaping is fragile.
"""
import io
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "results", "bearings", "onset_sensitivity.json")


def build():
    d = json.load(io.open(SRC, encoding="utf-8"))

    def cell(ds, thr, persist):
        #  the 80% sustained requirement barely moves anything, so the table
        #  fixes it at the value used throughout and sweeps the other two
        r = next(x for x in d[ds] if x["thr"] == thr and x["persist"] == persist
                 and x["frac"] == 0.8)
        m = r["median_onset_pct"]
        return (f"{m:.0f}" if m is not None else "--"), r["n_detected"], r["n_records"]

    THR = [2.0, 2.5, 3.0, 4.0, 5.0]
    PERSIST = [5, 10, 20, 40]
    rows = ""
    for persist in PERSIST:
        cells = []
        for ds in ("pronostia", "xjtu"):
            for thr in THR:
                m, nd, nr = cell(ds, thr, persist)
                cells.append(f"{m} ({nd})")
        mark = r"$\star$" if persist == 10 else ""
        rows += f"{persist}{mark} & " + " & ".join(cells) + r"\\" + "\n"

    hdr = " & ".join(f"${t:g}$" for t in THR)
    return f"""\\begin{{table}}[htbp]
\\centering\\small
\\caption{{Median detected onset (\\% of life) as the detector's constants vary, with the
number of records with an onset in parentheses; rows, persistence run; columns, band in robust
standard deviations; $\\star$, the setting used throughout.}}
\\label{{tab:onsetsens}}
\\begin{{adjustbox}}{{max width=\\linewidth}}
\\begin{{tabular}}{{@{{}}r rrrrr rrrrr@{{}}}}
\\toprule
& \\multicolumn{{5}}{{c}}{{PRONOSTIA}} & \\multicolumn{{5}}{{c}}{{XJTU-SY}}\\\\
\\cmidrule(lr){{2-6}}\\cmidrule(lr){{7-11}}
persist & {hdr} & {hdr}\\\\
\\midrule
{rows}\\bottomrule
\\end{{tabular}}
\\end{{adjustbox}}
\\end{{table}}
"""


if __name__ == "__main__":
    out = os.path.join(HERE, "..", "results", "tables", "tab_onsetsens.tex")
    io.open(out, "w", encoding="utf-8").write(build())
    print("  -> results/tables/tab_onsetsens.tex")
