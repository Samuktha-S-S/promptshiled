"""
PromptShield - Probability Calibration Module
Ensures classifier outputs are well-calibrated probabilities.
"""

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator
from typing import Optional
import joblib


def calibrate_classifier(
    classifier: BaseEstimator,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = "sigmoid"
) -> CalibratedClassifierCV:
    """
    Calibrate a classifier's probability outputs using validation data.
    
    For SVM and other classifiers that don't produce well-calibrated
    probabilities natively, this uses Platt scaling (sigmoid) or
    isotonic regression.
    """
    calibrated = CalibratedClassifierCV(
        estimator=classifier,
        method=method,
        cv=3
    )
    calibrated.fit(X_val, y_val)
    return calibrated


def evaluate_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> dict:
    """
    Evaluate probability calibration using Expected Calibration Error (ECE).
    
    Returns:
        Dictionary with calibration metrics
    """
    from sklearn.calibration import calibration_curve
    
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )
    
    # Expected Calibration Error
    bin_counts = np.histogram(y_prob, bins=n_bins, range=(0, 1))[0]
    total = len(y_true)
    ece = 0.0
    for i in range(len(fraction_of_positives)):
        if bin_counts[i] > 0:
            ece += (bin_counts[i] / total) * abs(
                fraction_of_positives[i] - mean_predicted_value[i]
            )
    
    return {
        "ece": float(ece),
        "fraction_of_positives": fraction_of_positives.tolist(),
        "mean_predicted_value": mean_predicted_value.tolist(),
        "n_bins": n_bins,
    }
