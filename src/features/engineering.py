"""
Shared feature-engineering module for P7.

KEYSTONE DESIGN RULE: this module is the SINGLE source of feature computation.
Both model training AND the live serving API import these exact functions, so
the features can never drift between training and production
(this is the fix for training-serving skew).

TRAILING WINDOWS ONLY: every rolling feature looks strictly BACKWARD. We never
use a reading from the future to compute a feature for the present, because in
live serving the future has not happened yet. This prevents data leakage and
keeps training honest about what serving can actually see.

Time-domain features (this file):
  - rolling mean   : the recent average  (is the baseline drifting up?)
  - rolling std    : recent volatility   (is the signal getting erratic?)
  - rolling min/max: the recent range
  - rate of change : movement vs the previous reading (sudden acceleration?)
"""

import numpy as np
import pandas as pd
from scipy.fft import rfft

DEFAULT_WINDOW = 20


def compute_time_domain_features(
    values: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Compute trailing-window time-domain features for one asset's readings.

    Args:
        values: time-ORDERED sensor values for a single asset (oldest first).
        window: size of the trailing window (number of past readings).

    Returns:
        A DataFrame with one row per input reading and columns:
        roll_mean, roll_std, roll_min, roll_max, rate_of_change.
        Early rows (fewer than `window` readings of history) are filled using
        the available history via min_periods=1, so no row is dropped.
    """
    values = pd.Series(values).reset_index(drop=True)

    # Trailing rolling window: min_periods=1 means early rows use whatever
    # history exists rather than producing NaN / being dropped.
    roll = values.rolling(window=window, min_periods=1)

    features = pd.DataFrame(
        {
            "roll_mean": roll.mean(),
            "roll_std": roll.std().fillna(0.0),  # std of a single value is NaN -> 0
            "roll_min": roll.min(),
            "roll_max": roll.max(),
            # diff() = current - previous; first row has no previous -> 0
            "rate_of_change": values.diff().fillna(0.0),
        }
    )

    return features


def _fft_summary(window_values: np.ndarray) -> tuple[float, float, float]:
    """Summarise one window's frequency spectrum into 3 features.

    Returns: (spectral_energy, dominant_frequency_index, spectral_entropy)
    """
    # Real FFT -> magnitude spectrum (how much energy at each frequency).
    spectrum = np.abs(rfft(window_values))

    # Total energy across all frequencies.
    spectral_energy = float(np.sum(spectrum**2))

    # Which frequency bin is loudest (the dominant frequency).
    dominant_freq = float(np.argmax(spectrum)) if len(spectrum) > 0 else 0.0

    # Spectral entropy: how spread-out the energy is.
    # Low entropy = energy concentrated in a sharp peak (possible fault).
    total = np.sum(spectrum)
    if total <= 0:
        spectral_entropy = 0.0
    else:
        p = spectrum / total  # normalise to a probability dist
        p = p[p > 0]  # ignore zero bins (log undefined)
        spectral_entropy = float(-np.sum(p * np.log(p)))

    return spectral_energy, dominant_freq, spectral_entropy


def compute_fft_features(
    values: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """Trailing-window FFT summary features, one row per reading.

    For each reading we FFT the trailing window ending at that reading
    (strictly backward -> no future leakage) and summarise the spectrum.
    """
    values = pd.Series(values).reset_index(drop=True).to_numpy()
    n = len(values)

    energy = np.zeros(n)
    dom = np.zeros(n)
    entropy = np.zeros(n)

    for i in range(n):
        start = max(0, i - window + 1)  # trailing window ending at i
        w = values[start : i + 1]
        energy[i], dom[i], entropy[i] = _fft_summary(w)

    return pd.DataFrame(
        {
            "fft_energy": energy,
            "fft_dominant_freq": dom,
            "fft_entropy": entropy,
        }
    )


def compute_features(
    values: pd.Series,
    window: int = DEFAULT_WINDOW,
) -> pd.DataFrame:
    """THE feature function: time-domain + FFT combined into one table.

    This is the single entry point that BOTH training and the serving API call,
    guaranteeing identical features in both places (no training-serving skew).
    """
    time_feats = compute_time_domain_features(values, window=window)
    fft_feats = compute_fft_features(values, window=window)
    return pd.concat([time_feats, fft_feats], axis=1)
