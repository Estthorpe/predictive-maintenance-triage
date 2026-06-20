# Phase 3 - Model Development: Findings

## Summary

Built a shared feature module (time-domain + FFT + per-asset baseline z-score)
and evaluated unsupervised (Isolation Forest) and supervised (LightGBM) anomaly
detection on both synthetic data and the real NASA IMS bearing dataset, judged
by the Phase-2 evaluation gate (recall@K, false-positive budget, beat-baseline).

The headline conclusion is honest and deliberate: **for run-to-failure bearing
data, a calibrated raw-RMS threshold is a strong baseline that the unsupervised
model does not clearly beat - because the physics makes vibration magnitude
genuinely predictive of failure, and because the model''s notion of "anomaly"
(spatial outlier) differs from the labels'' notion (end-of-life degradation).**
This is documented as a real finding, not hidden.

## What was built

- **Shared feature module** (`src/features/engineering.py`), used identically by
  training and serving (no training-serving skew):
  - Time-domain: rolling mean/std/min/max, rate-of-change (trailing windows only).
  - Frequency-domain: FFT energy, dominant frequency, spectral entropy.
  - Per-asset baseline z-score: deviation from an asset''s own healthy baseline.
- **Unsupervised model**: Isolation Forest wrapper (`src/models/anomaly.py`).
- **Supervised model**: LightGBM wrapper (`src/models/supervised.py`).
- **Real-data pipeline**: streamed NASA IMS 1st_test from the zip without
  extracting ~10 GB, summarised each 1-second 20 kHz file to per-bearing RMS,
  generated labels two ways (3-sigma control chart; time-based RUL window).

## Investigation log (what was tried and learned)

1. **Synthetic, simple spike anomalies** - Isolation Forest got recall 0.87 but
   LOST to the naive baseline (1.00): the spikes WERE the largest raw values, so
   a one-line threshold caught them. Lesson: a model must earn its complexity.

2. **Synthetic, realistic (drift + volatility) anomalies** - recall dropped to
   0.35; the model genuinely could not separate subtle pattern anomalies from
   normal. Diagnosed via score distributions (anomalies scored ~0.66 vs normal
   ~0.64 - almost no separation). Lesson: Isolation Forest finds spatial
   outliers, not subtle temporal patterns.

3. **Supervised LightGBM** - first attempt scored 0.13 because the time-based
   split put ALL anomalies in the test set (0 in training). Lesson: supervised
   models need examples of every class in training; unsupervised wants the
   opposite (clean normal-only training). Fixed the split -> 0.48.

4. **Anomaly-rate vs metric mismatch** - flooding the data with anomalies (37%
   of test set) broke recall@K@10% (ceiling of ~0.27). Lesson: the metric and
   the data distribution must agree - recall@K@10% needs anomalies < 10%.

5. **Real NASA bearing data** - Bearing 3 failed catastrophically (RMS tripled);
   Bearing 4 degraded subtly and noisily; Bearings 1-2 stayed healthy (matches
   the published IMS ground truth). Isolation Forest caught Bearing 3 perfectly
   (recall 1.00) - on real clear failures it works.

6. **Subtle case (Bearing 4)** - recall 0.48. Diagnosed: anomalies scored just
   below the top-10% cutoff (a near-miss of 0.010), crowded out by noisy normal
   readings. Score smoothing was tried and made it WORSE (0.48 -> 0.35) because
   the anomalies were scattered, not sustained - smoothing blurred the signal.
   Lesson: a fix must match the shape of the problem.

7. **Engineered the per-asset z-score feature** - lifted Bearing 4 from 0.48 to
   0.87 by measuring deviation from the bearing''s OWN baseline rather than
   absolute magnitude. A real feature-engineering win against a diagnosed gap.

8. **Discovered labelling bias** - the 3-sigma labels were themselves a raw-RMS
   rule, so a raw-RMS threshold was guaranteed to win (circular evaluation).
   Re-labelled using a bias-free TIME-BASED (RUL) rule.

9. **Fair test (time-based labels)** - the threshold still beat the model. Two
   honest reasons: (a) vibration magnitude is genuinely physically predictive of
   end-of-life, so RMS is a strong, legitimate signal; (b) Isolation Forest''s
   "anomaly = spatial outlier" does not match the labels'' "anomaly = end-of-life
   degradation" - a model/label definition mismatch.

## Conclusions

- On **clear failures**, a calibrated threshold suffices; a model is not needed.
- On **subtle failures**, a per-asset baseline-deviation feature meaningfully
  helps, but a global threshold remains hard to beat because vibration magnitude
  is physically tied to failure.
- Beating the threshold on a fair, time-based evaluation would require a model
  whose anomaly-definition matches the labels'' temporal definition - i.e. a
  **sequence model** (see Future Work).
- The evaluation gate worked exactly as designed throughout: it repeatedly
  refused to pass models that did not genuinely add value, preventing
  self-deception.

## Future work

- **Sequence model (LSTM/TCN)**: an order-aware model that learns the *trajectory*
  toward failure would match the time-based labels where Isolation Forest cannot.
  Deferred: out of scope for P7''s lifecycle focus and the local hardware budget
  (no GPU, limited disk).
- **Better ground-truth labels**: combine control-chart and RUL signals, or use
  the documented IMS failure annotations, to reduce labelling bias further.
- **Per-asset thresholds**: calibrate the flag threshold per bearing rather than
  globally.
