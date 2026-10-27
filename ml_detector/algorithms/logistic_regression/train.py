"""
PromptShield - Logistic Regression Training

Usage:
    python -m ml_detector.algorithms.logistic_regression.train

Trains on the shared training split (Sentence-BERT embeddings), selects the
ALLOW/FLAG/BLOCK thresholds on the validation split and saves
ml_detector/models/logistic_regression.joblib (+ .json metadata). The test split is not used.
"""

from sklearn.linear_model import LogisticRegression

from ml_detector.algorithms.common import train_and_save
from ml_detector.algorithms.logistic_regression import ALGORITHM
from ml_detector.algorithms.logistic_regression.predict import predict
from ml_detector.config import RANDOM_SEED

# class_weight="balanced": dataset v2 is 155 benign / 107 injection in train; reweighting keeps the
# extra benign conversational examples from shifting the boundary toward "benign".
HYPERPARAMETERS = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced"}


def build_model(seed: int = RANDOM_SEED) -> LogisticRegression:
    return LogisticRegression(random_state=seed, **HYPERPARAMETERS)


def main(models_dir=None) -> dict:
    return train_and_save(ALGORITHM, build_model, predict, HYPERPARAMETERS, models_dir=models_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"Train {ALGORITHM} on the shared splits.")
    parser.add_argument("--models-dir", default=None,
                        help="save to this directory instead of ml_detector/models (e.g. to stage a candidate)")
    main(parser.parse_args().models_dir)
