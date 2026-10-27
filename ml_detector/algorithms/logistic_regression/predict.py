"""
PromptShield - Logistic Regression Prediction

Usage:
    python -m ml_detector.algorithms.logistic_regression.predict "your prompt" ["another prompt" ...]

The score is the model's logistic output P(prompt_injection | embedding).
Logistic regression optimises log-loss, so this is a probability estimate,
but it has NOT been separately calibrated; its validation-set calibration
(ECE / Brier score) is recorded in the model metadata.
"""

import numpy as np

from ml_detector.algorithms.common import ClassifierOutput, injection_probability, load_artifact, predict_cli
from ml_detector.algorithms.logistic_regression import ALGORITHM

SCORE_TYPE = "logistic_probability"


def load(models_dir=None, embedding_info=None):
    """Return (model, metadata) for the saved Logistic Regression model."""
    return load_artifact(ALGORITHM, models_dir, embedding_info)


def predict(model, X: np.ndarray) -> ClassifierOutput:
    p_injection = injection_probability(model, X)
    return ClassifierOutput(
        labels=(p_injection >= 0.5).astype(int),
        p_injection=p_injection,
        score_type=SCORE_TYPE,
    )


if __name__ == "__main__":
    predict_cli(ALGORITHM, predict)
