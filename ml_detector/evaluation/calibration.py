"""
PromptShield - Probability Calibration Checks
Measures how well a classifier's injection score behaves as a probability.

A score is only described as a "calibrated probability" in this project when
it was produced by an explicit calibration step (Platt scaling for the Linear
SVM) AND has been measured here. With the small validation set these numbers
are noisy and should be read as indicative only.
"""

import numpy as np
from sklearn.metrics import brier_score_loss


def evaluate_calibration(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> dict:
    """
    Expected Calibration Error (ECE) and Brier score for P(prompt_injection).

    ECE = sum over bins of (bin_size / N) * |observed injection rate - mean predicted probability|
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_ids = np.minimum((y_prob * n_bins).astype(int), n_bins - 1)
    ece = 0.0
    bins = []
    for b in range(n_bins):
        mask = bin_ids == b
        if not mask.any():
            continue
        observed = float(y_true[mask].mean())
        predicted = float(y_prob[mask].mean())
        ece += (mask.sum() / len(y_true)) * abs(observed - predicted)
        bins.append({"bin": b, "count": int(mask.sum()), "observed": observed, "predicted": predicted})

    return {
        "samples": int(len(y_true)),
        "ece": float(ece),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "n_bins": n_bins,
        "bins": bins,
    }
