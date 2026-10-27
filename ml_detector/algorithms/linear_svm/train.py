"""
PromptShield - Linear SVM Training

Usage:
    python -m ml_detector.algorithms.linear_svm.train

A LinearSVC only produces a signed distance from its separating hyperplane.
To obtain a score the decision policy can threshold, it is wrapped in Platt
scaling (CalibratedClassifierCV, sigmoid, 3-fold CV on the TRAINING split
only). Calibration quality is measured on the validation split and stored
in the model metadata. The test split is not used.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from ml_detector.algorithms.common import train_and_save
from ml_detector.algorithms.linear_svm import ALGORITHM
from ml_detector.algorithms.linear_svm.predict import predict
from ml_detector.config import RANDOM_SEED

# class_weight="balanced": compensates for the 155/107 class imbalance of dataset v2.
SVM_HYPERPARAMETERS = {"C": 1.0, "max_iter": 2000, "class_weight": "balanced"}
CALIBRATION = {"method": "sigmoid", "cv": 3}
HYPERPARAMETERS = {**SVM_HYPERPARAMETERS, "calibration": CALIBRATION}


def build_model(seed: int = RANDOM_SEED) -> CalibratedClassifierCV:
    return CalibratedClassifierCV(
        estimator=LinearSVC(random_state=seed, **SVM_HYPERPARAMETERS),
        **CALIBRATION,
    )


def main(models_dir=None) -> dict:
    return train_and_save(ALGORITHM, build_model, predict, HYPERPARAMETERS, models_dir=models_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=f"Train {ALGORITHM} on the shared splits.")
    parser.add_argument("--models-dir", default=None,
                        help="save to this directory instead of ml_detector/models (e.g. to stage a candidate)")
    main(parser.parse_args().models_dir)
