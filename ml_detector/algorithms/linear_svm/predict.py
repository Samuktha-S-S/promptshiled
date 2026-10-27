"""
PromptShield - Linear SVM Prediction

Usage:
    python -m ml_detector.algorithms.linear_svm.predict "your prompt" ["another prompt" ...]

Two scores are returned:
  * decision_scores - the raw SVM margin (mean over the 3 cross-validation
    SVMs). Positive = injection side of the hyperplane. This is a distance,
    NOT a probability.
  * p_injection - the Platt-scaled (sigmoid-calibrated) probability, which is
    what the decision policy thresholds.
"""

import numpy as np

from ml_detector.algorithms.common import (
    ClassifierOutput, injection_column, injection_probability, load_artifact, predict_cli,
)
from ml_detector.algorithms.linear_svm import ALGORITHM

SCORE_TYPE = "platt_calibrated_probability"


def load(models_dir=None, embedding_info=None):
    """Return (model, metadata) for the saved Linear SVM model."""
    return load_artifact(ALGORITHM, models_dir, embedding_info)


def decision_scores(model, X: np.ndarray) -> np.ndarray:
    """
    Raw signed margin, averaged over the calibrated folds' LinearSVCs, oriented so
    that positive always means the prompt_injection side (sklearn's decision_function
    is positive for classes_[1], so the sign is flipped if the class order differs).
    """
    margins = []
    for calibrated in model.calibrated_classifiers_:
        svm = calibrated.estimator
        sign = 1.0 if injection_column(svm) == 1 else -1.0
        margins.append(sign * svm.decision_function(X))
    return np.mean(margins, axis=0)


def predict(model, X: np.ndarray) -> ClassifierOutput:
    p_injection = injection_probability(model, X)
    return ClassifierOutput(
        labels=(p_injection >= 0.5).astype(int),
        p_injection=p_injection,
        score_type=SCORE_TYPE,
        decision_scores=decision_scores(model, X),
    )


if __name__ == "__main__":
    predict_cli(ALGORITHM, predict)
