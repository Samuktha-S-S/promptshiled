"""
PromptShield - Random Forest Training

Usage:
    python -m ml_detector.algorithms.random_forest.train

Trains on the shared training split (Sentence-BERT embeddings), selects the
ALLOW/FLAG/BLOCK thresholds on the validation split and saves
ml_detector/models/random_forest.joblib (+ .json metadata). The test split is not used.
"""

from sklearn.ensemble import RandomForestClassifier

from ml_detector.algorithms.common import train_and_save
from ml_detector.algorithms.random_forest import ALGORITHM
from ml_detector.algorithms.random_forest.predict import predict
from ml_detector.config import RANDOM_SEED

# class_weight="balanced": compensates for the 155/107 class imbalance of dataset v2.
HYPERPARAMETERS = {"n_estimators": 200, "max_depth": 20, "n_jobs": -1, "class_weight": "balanced"}


def build_model(seed: int = RANDOM_SEED) -> RandomForestClassifier:
    return RandomForestClassifier(random_state=seed, **HYPERPARAMETERS)


def main(models_dir=None) -> dict:
    return train_and_save(ALGORITHM, build_model, predict, HYPERPARAMETERS, models_dir=models_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"Train {ALGORITHM} on the shared splits.")
    parser.add_argument("--models-dir", default=None,
                        help="save to this directory instead of ml_detector/models (e.g. to stage a candidate)")
    main(parser.parse_args().models_dir)
