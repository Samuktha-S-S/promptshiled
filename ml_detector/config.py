"""
PromptShield - ML Detector Configuration
Shared paths, labels and seeds used by every algorithm, so that training,
evaluation and inference agree on the same conventions.
"""

import os

PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS_DIR = os.path.join(PACKAGE_DIR, "datasets")
RAW_DIR = os.path.join(DATASETS_DIR, "raw")
SPLITS_DIR = os.path.join(DATASETS_DIR, "splits")
CHALLENGE_DIR = os.path.join(DATASETS_DIR, "challenge")
PROCESSED_DIR = os.path.join(DATASETS_DIR, "processed")

MODELS_DIR = os.path.join(PACKAGE_DIR, "models")
PRODUCTION_CONFIG_FILE = "production_model.json"

EVALUATION_DIR = os.path.join(PACKAGE_DIR, "evaluation")
RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")
FIGURES_DIR = os.path.join(EVALUATION_DIR, "figures")

# Fixed label order: 0 = benign, 1 = prompt_injection (positive class)
LABELS = ("benign", "prompt_injection")
POSITIVE_LABEL = "prompt_injection"

RANDOM_SEED = 42

ALGORITHMS = ("logistic_regression", "linear_svm", "random_forest")
