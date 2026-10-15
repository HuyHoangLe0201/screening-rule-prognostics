# Pre-registered test of the step across the threshold (paper 3)

Written and frozen before any remaining-life prediction was computed on the
units below.

## Why this test

The blind test (`PREREG_p3_blind.md`) had a single unit with onset after 70 %
of life, so it could not test whether prediction worsens across the threshold.
This protocol draws a cohort designed to contain early, middle and late onsets,
and tests the step on units whose remaining life has never been predicted.

## What is already known

* The detected onsets of the C-MAPSS units were computed for the screen
  (`study_thirdset.py`); the cohort below is drawn using them.
* The step was found on FD002 and FD004 units sampled by onset
  (`study_bandtest.py`, two draws stored in `bandtest.json` and
  `bandtest_rep.json`).
* No unit eligible below has had its remaining life predicted by any model.

## Units

* Pool: every unit of C-MAPSS FD001 and FD003 (raw indicator) and FD002 and
  FD004 (normalised within operating regime), as `study_thirdset.load` and
  `indicator` build them, with a detected onset and at least 60 cycles,
  excluding every unit that appears in `bandtest.json` or `bandtest_rep.json`.
* Onset bins: [0, 40), [40, 55), [55, 70), [70, 80), [80, 90) and [90, 101) %
  of life.  From each bin, 20 units are drawn without replacement with
  `numpy.random.default_rng(20261020).permutation`, applied to the bins in that
  order over the pool sorted by key; a bin with fewer than 20 units contributes
  all of them.

## Predictors

* Path model: exactly as in `study_bandtest.py`, that is,
  `study_bearings.predict_one` with law "BS", the informed prior, the Paris
  coupling and 600 particles, with the seed the blake2b hash of (unit key, k)
  modulo 10 000.
* D: the median, over the other units of the same subset, of the median of
  each unit's last three indicator values.
* `med_life_steps`: the median, over the other units of the same subset, of
  max(n - onset, 20).
* Life table: `study_p3_review._life_table` over the other units of the same
  subset, with lives n - 1 cycles.
* Gradient boosting, reported descriptively with no prediction attached.
  * Features at index k: k, h_k, the mean of h over the last 5 cycles, the
    least-squares slopes of h over the last 10 and the last 30 cycles, and the
    maximum of h so far.
  * Target: log of the remaining life.
  * Training: on the other units of the same subset, at every fifth index from
    20 to n - 2, in five folds by unit (seed 20261020).
  * Models: `sklearn.ensemble.GradientBoostingRegressor` with 200 trees, depth
    3, learning rate 0.05 and random_state 0, for the median; quantile models
    at 0.025 and 0.975 give the interval.
* Prediction points: k = int(f n) for f in {0.50, 0.70, 0.85}, skipping k < 20
  or k >= n - 1.  True RUL = (n - 1) - k cycles.

## Predictions

Onset groups: late means onset fraction 0.70 or more; early means below 0.70.

* D1 (primary): the path model's median absolute percentage error is higher in
  the late group than in the early group.  The bootstrap interval of the
  difference (late minus early), with units resampled within each group (10 000
  resamples, seed 20261020), lies wholly above 0.
* D2: the Spearman correlation between onset fraction and the unit's mean
  absolute percentage error is positive, with its bootstrap interval over units
  wholly above 0.
* D3: the path model's coverage of the 95 % interval is lower in the late group
  than in the early group, with the interval of the difference wholly below 0.
* D4: in the late group the path model does not beat the life table; the
  interval of the difference in median absolute percentage error (path model
  minus life table) does not lie wholly below 0.

## Reporting

Every outcome is reported whatever it is.  A failed D1 counts against the
paper's claim that a late onset marks worse prediction.  Any deviation is
reported as such.  `study_p3_step.py` refuses to run if this file's SHA-256
differs from the value recorded in it.
