# Pre-registered test of the screening rule on two archives it has never seen

Frozen on 2026-10-12, before any onset, indicator value or capacity curve of
either archive was computed or plotted.  Up to this point only the file
listings, the HDF5/MAT field names, the per-cycle description strings and the
datasets' own documentation were read.  The analysis script
(`code/study_p3_prereg.py`) is written after this file and stores this file's
SHA-256 in its output.  Nothing below may be changed after the analysis runs; a
prediction that fails is reported as failed.

## Archives

A. **XJTU battery dataset** (Wang et al., Nature Communications 15, 2024,
   doi:10.1038/s41467-024-48779-z).  55 LiNi0.5Co0.2Mn0.3O2 18650 cells in six
   batches, all cycled to end of life (capacity below 80 % of initial).  Local
   copy: `data/NASABattery/Battery Dataset/Batch-1 ... Batch-6`.  All 55 cells
   are used.

B. **N-CMAPSS** (Arias Chao et al., Data 6(1):5, 2021), the two files already
   extracted on disk: `N-CMAPSS_DS01-005.h5` and `N-CMAPSS_DS02-006.h5`.  All
   units of both files, development and test splits together (the screen is
   unsupervised).  The other subsets sit inside a 16 GB nested archive that was
   not extracted; they are excluded in advance, not after looking.

## Indicator (fixed now)

A. One capacity value per measurement, in the order measured, from
   `summary.discharge_capacity_Ah`, using only cycles whose description shows a
   capacity at a constant reference rate:
   * Batch-1, Batch-2: every regular cycle (fixed 1C discharge to 2.5 V);
   * Batch-3: the cycles described "2C charge and 1C discharge" (the discharge
     rate cycles through 0.5-5C, so only the 1C cycles measure capacity on one
     scale);
   * Batch-4, Batch-5, Batch-6: the cycles described "[test capacity]".
   In every batch the first, initial-capacity cycle (0.2C discharge, a
   different protocol) is dropped.  The indicator is the capacity drop from the
   first retained value, `hi = cap[0] - cap`, exactly the construction used for
   the NASA cells in `study_battery.py`.  Life is the length of the retained
   series; onset is reported as index / length.

B. Per unit and per flight cycle: each of the 14 sensors `X_s` is regressed on
   a full quadratic in the four standardised scenario descriptors `W` (alt,
   Mach, TRA, T2; 15 terms with intercept), fitted by least squares on the
   per-second data of the unit's healthy-window cycles, the first
   `min(40, max(5, n // 10))` of its `n` cycles.  The cycle mean of each
   sensor's residual gives an `n x 14` matrix, which goes unchanged into
   `study_thirdset.indicator` and `study_thirdset.onset`.  This is step 1 of the
   paper's Algorithm 3 for continuously varying operating conditions.  The
   dataset's health-state label `hs` gives a labelled transition: the first
   cycle with `hs = 0`.

## Detector and rule (the paper's, unchanged)

Band 3 robust s.d., persistence 10, sustained-departure fraction 0.8, healthy
window `min(40, max(5, n // 10))`.  Algorithm 3: an onset in fewer than a
quarter of the records means "check indicator"; otherwise a censored median
(undetected records right-censored at the end of life) at 70 % of life or later
declines the first-passage model; otherwise "not declined".  Intervals: Wilson
for the detection rate, percentile bootstrap with 10,000 resamples and seed
20261015 for the medians.

## Predictions

* **P1** (batteries): detection rate at least 0.25 and censored median below
  70 %, so "not declined".  Basis: capacity fades from the first cycle, as in
  the NASA cells.
* **P2** (N-CMAPSS): detection rate at least 0.25 and censored median below
  70 %, so "not declined".  Basis: the C-MAPSS subsets at 56-68 %.
* **P3** (no archive in the band): neither archive's censored median lies
  between 70 and 90 % of life.
* **P4** (the detector against a labelled transition, N-CMAPSS only): in at
  least three quarters of the units with a detected onset, the detected onset is
  at or after the labelled transition.  The median lag, detected minus
  labelled, as a fraction of life, is reported without a prediction.

## Reporting

All four outcomes go into the paper whichever way they fall, with the per-record
onsets in `results/bearings/p3_prereg.json`.  No constant is tuned on these
archives, and no record is dropped after the analysis except for a load failure,
which is reported with its cause.
