# -*- coding: utf-8 -*-
r"""One house style for every figure, and the right size for each.

Two things were wrong with the figures, and the second is the one that made
them look cheap.

The typeface did not match the paper.  The manuscripts are set in Latin Modern
and the figures were drawn in DejaVu Sans, so every axis label announced itself
as an import.  TeX Live ships the Latin Modern OpenType files; registering them
with matplotlib makes figure text the same face as body text, and setting the
maths font to Computer Modern matches the equations too.

The figures were drawn at the wrong size.  \linewidth here is 390pt = 5.40in,
and the figures were drawn 6.6-6.8in wide and then scaled to fit, so a 9pt label
printed at about 7.2pt.  The four inside subfigures were far worse: a panel
declared 0.36\linewidth holds a figure drawn at 3.9in, a scale factor of one
half, so its 9pt labels reached the page at 4.5pt -- below the minimum any
journal accepts.  Drawing each figure at exactly the width it will occupy fixes
this, and it is why `width_of` exists.

For the size to survive, the saved file must be exactly the requested figure
size: `bbox_inches="tight"` crops to the ink, which shrinks the file and lets
\includegraphics scale it back up by an unpredictable factor.  So layout is
constrained and `save` writes without cropping.

    import figstyle
    figstyle.apply()
    fig, ax = plt.subplots(1, 2, figsize=figstyle.size("fig_grid", 0.42))
"""
from __future__ import annotations

import glob
import os

import matplotlib
import matplotlib.font_manager as fm

#  \linewidth of the manuscripts, measured with \the\linewidth
COLUMN_PT = 390.0
COLUMN_IN = COLUMN_PT / 72.27

#  Fraction of \linewidth each figure actually occupies on the page, including
#  the subfigure it sits in.  Kept here rather than in the drawing code so that
#  a change to the layout is a one-line change in one file.
FRACTION = {
    "fig_grid": 1.00,
    "fig_lamperti": 1.00,
    "fig_theory_derivation": 0.48,
    "fig_theory_igapprox": 0.92,
    "fig_head": 1.00,
    "fig_calibration": 1.00,
    "fig_pit": 1.00,
    "fig_guided": 1.00,
    "fig_onset": 1.00,
    "fig_theory_pfconv": 0.36,
    "fig_theory_epistemic": 0.62,
    "fig_theory_robust": 0.47,
    "fig_theory_prior": 0.47,
    "fig_realdata": 1.00,
}

#  One palette, used by every figure.  Dark blue is always the proposed or
#  corrected object, red always the thing it is being compared against, grey
#  always a reference the reader should not read as a competitor.
C = {
    "gbs": "#1f4e79",     # the proposed construction
    "ig": "#c0504d",      # the comparison it is measured against
    "ref": "#7f7f7f",     # a reference line or a null
    "alt": "#4f8f3f",     # a third series
    "acc": "#e08214",     # an accent for annotation
    "band": "#c8d6e5",    # a filled uncertainty band behind a line
    "faint": "#d9d9d9",   # individual records behind a summary
}

HEADC = {"BS": "#1f4e79", "BS-t": "#6a9bd1", "Weibull": "#4f8f3f",
         "LogNormal": "#e08214", "IGauss": "#c0504d", "Gaussian": "#7f7f7f"}
HEADL = {"BS": "BS", "BS-t": "BS-$t$", "Weibull": "Weibull",
         "LogNormal": "lognormal", "IGauss": "inverse Gaussian",
         "Gaussian": "Gaussian"}


def _register_latin_modern() -> str:
    """Give matplotlib the typeface the manuscripts are set in, if TeX Live has it."""
    for root in ("D:/texlive/2026/texmf-dist/fonts/opentype/public/lm",
                 "/d/texlive/2026/texmf-dist/fonts/opentype/public/lm",
                 "/usr/share/texmf/fonts/opentype/public/lm"):
        hits = sorted(set(glob.glob(os.path.join(root, "lmroman10-*.otf"))))
        if not hits:
            continue
        for f in hits:
            try:
                fm.fontManager.addfont(f)
            except Exception:
                pass
        names = {x.name for x in fm.fontManager.ttflist}
        if "Latin Modern Roman" in names:
            return "Latin Modern Roman"
    #  STIX is bundled with matplotlib and is a serif; DejaVu Serif is the
    #  last resort.  Either is closer to the page than a sans face.
    names = {x.name for x in fm.fontManager.ttflist}
    for cand in ("STIXGeneral", "DejaVu Serif"):
        if cand in names:
            return cand
    return "serif"


def apply() -> None:
    face = _register_latin_modern()
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": [face],
        "mathtext.fontset": "cm",       # Computer Modern, to match lmodern

        #  Drawn at printed size, so these are the sizes the reader sees.
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8.5,
        "legend.fontsize": 7.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,

        #  A quieter frame: the data should be the darkest thing on the page.
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "axes.labelcolor": "#222222",
        "text.color": "#222222",
        "xtick.color": "#444444",
        "ytick.color": "#444444",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.direction": "out",
        "ytick.direction": "out",

        "axes.grid": True,
        "grid.color": "#000000",
        "grid.alpha": 0.10,
        "grid.linewidth": 0.4,
        "axes.axisbelow": True,

        "lines.linewidth": 1.2,
        "lines.markersize": 3.6,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.handletextpad": 0.5,
        "legend.columnspacing": 1.2,
        "legend.borderaxespad": 0.3,

        "figure.dpi": 150,
        "savefig.dpi": 600,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.02,
        "figure.constrained_layout.w_pad": 0.02,
        "figure.constrained_layout.hspace": 0.03,
        "figure.constrained_layout.wspace": 0.03,
        "pdf.fonttype": 42,             # embed as TrueType, not Type 3
        "ps.fonttype": 42,
    })


def width_of(name: str) -> float:
    """Printed width of this figure, in inches."""
    if name not in FRACTION:
        raise KeyError(f"{name}: add it to figstyle.FRACTION with the fraction "
                       f"of \\linewidth it occupies on the page")
    return COLUMN_IN * FRACTION[name]


def size(name: str, aspect: float) -> tuple:
    """(width, height) in inches, with height given as a fraction of width."""
    w = width_of(name)
    return (w, w * aspect)


def save(fig, name: str, fig_dir: str) -> None:
    """Write the figure at exactly its declared size.

    No bbox cropping: the point of drawing at printed size is that
    \\includegraphics does not have to scale, and a cropped file would be
    scaled back up by whatever fraction the crop removed.
    """
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(fig_dir, f"{name}.{ext}"), bbox_inches=None)
    import matplotlib.pyplot as plt
    plt.close(fig)
