"""
PromptShield - Random Forest Prediction

Usage:
    python -m ml_detector.algorithms.random_forest.predict "your prompt" ["another prompt" ...]

The score is RandomForestClassifier.predict_proba: the mean over the 200
trees of each leaf's injection fraction (close to the share of trees voting
'injection'). It is an ensemble vote, not a calibrated probability; its
validation-set ECE / Brier score is recorded in the model metadata.
"""

import numpy as np

from ml_detector.algorithms.common import ClassifierOutput, injection_probability, load_artifact, predict_cli
from ml_detector.algorithms.random_forest import ALGORITHM

SCORE_TYPE = "tree_vote_fraction"


def load(models_dir=None, embedding_info=None):
    """Return (model, metadata) for the saved Random Forest model."""
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
