# Pre-registered test: does a declined cohort predict worse? (paper 3)

Written and frozen before any remaining-useful-life prediction, error or
coverage was computed on any of the cells below.

## Question

Paper 3's screen (Algorithm 3) declines a first-passage model on an indicator
when the bootstrap interval of a cohort's censored median onset lies at or after
70 % of life.  The paper's fourth claim is that a pair the screen declines
predicts worse than one it passes.  That claim has so far been tested on one
asset class, the C-MAPSS turbofan units.  This protocol tests it on
lithium-ion cells, with every constant of the screen and of the predictors fixed
in advance.

## What is already known

Stated here so that none of it can be mistaken for a prediction.

* The onsets of the 55 XJTU cells were computed under
  `PREREG_p3_new_archives.md` and are stored in `results/bearings/p3_prereg.json`.
* Applying Algorithm 3 to each charging regime alone (the exploratory part
  `regime` of `study_p3_review.py`) declines the 3C cells (Batch-2, 15 cells,
  censored median 80.8 % [76.2, 82.5]), is undecided on the random-walk cells
  (Batch-5, 8 cells, 3 with an onset), and does not decline Batch-1 (2C),
  Batch-3 (R2.5), Batch-4 (R3) or Batch-6 (Sim_satellite).
* No remaining-useful-life prediction, error or coverage has been computed on
  any XJTU cell.

## Data and cohorts

* Cells: all 55, with the capacity series exactly as
  `study_p3_prereg.battery_capacity` returns it.  Indicator h_k = c_0 - c_k.
  Life is n - 1 cycles, the last cycle of the record.
* Declined cohort: Batch-2 (3C), 15 cells.
* Passed cohort: Batch-1, Batch-3, Batch-4 and Batch-6 pooled, 32 cells.
* Batch-5 (undecided) is reported descriptively only and enters no prediction.

## Predictors

Nothing is tuned on these cells.

* Path model: `study_bearings.predict_one` with law "BS", the informed prior,
  the Paris coupling and 600 particles.  The seed of each prediction is the
  blake2b hash of (cell, k), reduced modulo 10 000, as in
  `study_bandbattery.stable_seed`.
* Failure threshold D: the median, over the other cells of the same regime, of
  the median of each cell's last three indicator values.  This is the
  indicator-scale threshold of the C-MAPSS band study.
* `med_life_steps`: the median, over the other cells of the same regime, of
  max(n - onset_idx, 20).  onset_idx is taken from `p3_prereg.json`, and is n
  when no onset was detected.
* Life table: `study_p3_review._life_table` over the other cells of the same
  regime, with lives n - 1.  It gives the median and the 2.5 and 97.5 %
  quantiles of the residual life.
* Prediction points: k = int(f n) for f in {0.50, 0.70, 0.85}.  A point is
  skipped if k < 20 or k >= n - 1.  True RUL = (n - 1) - k cycles.

## Metrics

* Absolute percentage error |pred - true| / true x 100, and coverage of the
  95 % interval (lo <= true <= hi).
* Cohort statistics: the median absolute percentage error over the cohort's
  predictions, and coverage as the share of its predictions covered.
* Uncertainty: a bootstrap clustered on the cell, with cells resampled within
  each cohort, 10 000 resamples, seed 20261016, and percentile 95 % intervals.

## Predictions

* R1 (primary): the path model's median absolute percentage error is higher on
  the declined cohort than on the passed cohort, and the clustered-bootstrap
  95 % interval of the difference (declined minus passed) lies wholly above 0.
* R2: the path model's coverage is lower on the declined cohort than on the
  passed cohort, and the interval of the difference lies wholly below 0.
* R3: on the declined cohort the path model does not beat the life table.  The
  interval of the difference in median absolute percentage error (path model
  minus life table) does not lie wholly below 0.

## Reporting

All three outcomes are reported whatever they are, together with per-regime
results including Batch-5.  A failed R1 counts against the paper's fourth
claim.  Any deviation from this protocol is reported as a deviation.  The
analysis script `study_p3_rul.py` refuses to run if this file's SHA-256 differs
from the value recorded in the script.
