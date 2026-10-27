"""
PromptShield - Random Forest Evaluation

Usage:
    python -m ml_detector.algorithms.random_forest.evaluate

Evaluates the saved model on the held-out test split and writes
ml_detector/evaluation/results/random_forest_test_evaluation.json
and ml_detector/evaluation/figures/cm_random_forest.png.
"""

from ml_detector.algorithms.common import evaluate_saved, print_evaluation
from ml_detector.algorithms.random_forest import ALGORITHM
from ml_detector.algorithms.random_forest.predict import predict


def evaluate(models_dir=None, splits=None, save=True) -> dict:
    return evaluate_saved(ALGORITHM, predict, models_dir=models_dir, splits=splits, save=save)


if __name__ == "__main__":
    print_evaluation(evaluate())
