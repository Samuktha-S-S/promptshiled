"""
PromptShield - Shared Algorithm Infrastructure
Code that is identical for Logistic Regression, Linear SVM and Random Forest:
artifact storage, compatibility checks, and the train / evaluate drivers.

Only the classifier itself differs between algorithms; each algorithm folder
supplies its own `build_model()` (train.py) and `predict()` (predict.py) and
hands them to the drivers below. This keeps the experimental pipeline the
same for all three, which is what makes the comparison fair.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, Optional, Tuple

import joblib
import numpy as np
import sklearn

from ml_detector.config import FIGURES_DIR, LABELS, MODELS_DIR, RANDOM_SEED, RESULTS_DIR

ARTIFACT_FORMAT_VERSION = 1


class ArtifactError(RuntimeError):
    """Raised when a trained model artifact is missing, corrupt or incompatible."""


@dataclass
class ClassifierOutput:
    """Predictions for a batch of embeddings, in a form shared by all algorithms."""
    labels: np.ndarray                      # int labels, 0 = benign, 1 = prompt_injection
    p_injection: np.ndarray                 # score in [0, 1] used by the decision policy
    score_type: str                         # what p_injection actually is (see each README)
    decision_scores: Optional[np.ndarray] = None  # raw SVM margin; NOT a probability


INJECTION_CLASS = 1  # encoded label of "prompt_injection" (LABELS[1])


def injection_column(model) -> int:
    """
    Index of the prompt_injection column in predict_proba output, taken from the
    model's own `classes_` instead of assuming the column order.
    """
    classes = [int(c) for c in getattr(model, "classes_", [])]
    if sorted(classes) != [0, 1]:
        raise ArtifactError(f"Model classes {classes} are not the expected binary labels [0, 1].")
    return classes.index(INJECTION_CLASS)


def injection_probability(model, X: np.ndarray) -> np.ndarray:
    """P(prompt_injection) for each row, using the model's actual class order."""
    return model.predict_proba(X)[:, injection_column(model)]


def artifact_paths(algorithm: str, models_dir: Optional[str] = None) -> Tuple[str, str]:
    models_dir = models_dir or MODELS_DIR
    return (os.path.join(models_dir, f"{algorithm}.joblib"),
            os.path.join(models_dir, f"{algorithm}.json"))


def save_artifact(algorithm: str, model, metadata: dict, models_dir: Optional[str] = None) -> str:
    model_path, meta_path = artifact_paths(algorithm, models_dir)
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    return model_path


def load_artifact(algorithm: str, models_dir: Optional[str] = None, embedding_info: Optional[dict] = None):
    """
    Load a trained model and its metadata, refusing incompatible artifacts.

    Args:
        embedding_info: info() of the embedder that will be used at inference;
            if given, it must match the embedder the model was trained with.
    """
    model_path, meta_path = artifact_paths(algorithm, models_dir)
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        raise ArtifactError(
            f"No trained {algorithm} model in {os.path.dirname(model_path)}. "
            f"Run 'python -m ml_detector.algorithms.{algorithm}.train' first."
        )
    try:
        with open(meta_path) as f:
            metadata = json.load(f)
        model = joblib.load(model_path)
    except Exception as e:
        raise ArtifactError(f"Could not read {algorithm} artifact: {type(e).__name__}: {e}") from e

    if metadata.get("algorithm") != algorithm:
        raise ArtifactError(f"{meta_path} describes '{metadata.get('algorithm')}', not '{algorithm}'.")
    if metadata.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ArtifactError(f"{meta_path} has unsupported format_version {metadata.get('format_version')}.")
    if list(metadata.get("labels", [])) != list(LABELS):
        raise ArtifactError(f"{meta_path} label order {metadata.get('labels')} != {list(LABELS)}.")
    if embedding_info is not None and metadata.get("embedding") != embedding_info:
        raise ArtifactError(
            f"{algorithm} was trained with embedding {metadata.get('embedding')} but inference "
            f"uses {embedding_info}. Retrain the model."
        )
    injection_column(model)  # raises ArtifactError if the classes are not {0, 1}
    n_features = getattr(model, "n_features_in_", None)
    expected_dim = metadata.get("embedding", {}).get("embedding_dim")
    if n_features is not None and expected_dim is not None and n_features != expected_dim:
        raise ArtifactError(f"{algorithm} expects {n_features} features, metadata says {expected_dim}.")
    return model, metadata


def train_and_save(
    algorithm: str,
    build_model: Callable[[int], object],
    predict: Callable[[object, np.ndarray], ClassifierOutput],
    hyperparameters: dict,
    seed: int = RANDOM_SEED,
    models_dir: Optional[str] = None,
    splits: Optional[Dict] = None,
    embedder=None,
) -> dict:
    """
    Train one algorithm on the shared training split, select its decision
    thresholds on the validation split, and save the artifact.

    The test split is NOT touched here.
    """
    from ml_detector.datasets.loader import load_embedded_splits
    from ml_detector.embeddings import get_embedder
    from ml_detector.evaluation.calibration import evaluate_calibration
    from ml_detector.evaluation.metrics import classification_metrics
    from ml_detector.inference.decision_policy import select_thresholds

    embedder = embedder or get_embedder()
    splits = splits or load_embedded_splits(embedder)
    train, val = splits["train"], splits["val"]

    if len(np.unique(train.y)) != 2:
        raise ValueError("Training data must contain both benign and prompt_injection examples.")

    model = build_model(seed)
    start = time.perf_counter()
    model.fit(train.X, train.y)
    train_seconds = time.perf_counter() - start

    val_out = predict(model, val.X)
    thresholds = select_thresholds(val_out.p_injection, val.y)

    metadata = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "algorithm": algorithm,
        "hyperparameters": hyperparameters,
        "random_seed": seed,
        "labels": list(LABELS),
        "embedding": embedder.info(),
        "score_type": val_out.score_type,
        "thresholds": thresholds,
        "training": {
            "train_samples": int(len(train.y)),
            "validation_samples": int(len(val.y)),
            "train_seconds": round(train_seconds, 4),
            "train_accuracy": float((predict(model, train.X).labels == train.y).mean()),
        },
        "validation_metrics": classification_metrics(val.y, val_out.labels),
        "validation_calibration": evaluate_calibration(val.y, val_out.p_injection),
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(),
    }
    path = save_artifact(algorithm, model, metadata, models_dir)

    vm = metadata["validation_metrics"]
    print(f"[{algorithm}] trained on {len(train.y)} samples in {train_seconds:.2f}s -> {path}")
    print(f"[{algorithm}] validation: acc={vm['accuracy']:.4f} P={vm['precision']:.4f} "
          f"R={vm['recall']:.4f} F1={vm['f1']:.4f} FPR={vm['false_positive_rate']:.4f}")
    print(f"[{algorithm}] thresholds (validation-selected): "
          f"allow>={thresholds['benign_allow_threshold']}, block>={thresholds['injection_block_threshold']}")
    return metadata


def evaluate_saved(
    algorithm: str,
    predict: Callable[[object, np.ndarray], ClassifierOutput],
    models_dir: Optional[str] = None,
    splits: Optional[Dict] = None,
    embedder=None,
    save: bool = True,
) -> dict:
    """Evaluate a saved model on the held-out test split."""
    from ml_detector.datasets.loader import load_embedded_splits
    from ml_detector.embeddings import get_embedder
    from ml_detector.evaluation.metrics import (
        classification_metrics, decision_breakdown, per_category_metrics, plot_confusion_matrix,
    )
    from ml_detector.inference.decision_policy import DecisionPolicy

    embedder = embedder or get_embedder()
    model, metadata = load_artifact(algorithm, models_dir, embedder.info())
    splits = splits or load_embedded_splits(embedder)
    test = splits["test"]

    out = predict(model, test.X)
    policy = DecisionPolicy(metadata["thresholds"])
    actions = [
        policy.decide(LABELS[label], float(p if label == 1 else 1 - p)).action
        for label, p in zip(out.labels, out.p_injection)
    ]

    results = {
        "algorithm": algorithm,
        "split": "test",
        "metrics": classification_metrics(test.y, out.labels),
        "per_category": per_category_metrics(test.frame, test.y, out.labels),
        "decisions": decision_breakdown(actions, test.y),
        "thresholds": metadata["thresholds"],
        "score_type": metadata["score_type"],
    }

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        with open(os.path.join(RESULTS_DIR, f"{algorithm}_test_evaluation.json"), "w") as f:
            json.dump(results, f, indent=2)
        plot_confusion_matrix(results["metrics"]["confusion_matrix"],
                              f"{algorithm.replace('_', ' ').title()} (test)",
                              os.path.join(FIGURES_DIR, f"cm_{algorithm}.png"))
    return results


def predict_cli(algorithm: str, predict: Callable[[object, np.ndarray], ClassifierOutput], argv=None) -> None:
    """`python -m ml_detector.algorithms.<algorithm>.predict "prompt" ...`"""
    import argparse
    from ml_detector.embeddings import get_embedder

    parser = argparse.ArgumentParser(description=f"Classify prompts with the trained {algorithm} model.")
    parser.add_argument("prompts", nargs="+", help="one or more prompts to classify")
    parser.add_argument("--models-dir", default=None)
    args = parser.parse_args(argv)

    embedder = get_embedder()
    model, metadata = load_artifact(algorithm, args.models_dir, embedder.info())
    out = predict(model, embedder.encode(args.prompts))
    for i, prompt in enumerate(args.prompts):
        line = f"{LABELS[out.labels[i]]:<17} {metadata['score_type']}={out.p_injection[i]:.4f}"
        if out.decision_scores is not None:
            line += f"  svm_margin={out.decision_scores[i]:+.4f}"
        print(f"{line}  | {prompt}")


def print_evaluation(results: dict) -> None:
    m = results["metrics"]
    print(f"\n=== {results['algorithm']} - held-out test set ({m['samples']} prompts) ===")
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"):
        print(f"  {key:<22} {m[key]:.4f}")
    print(f"  confusion matrix [[TN FP] [FN TP]]: {m['confusion_matrix']}")
    print("\n  Per category (correct / samples):")
    for cat, r in results["per_category"].items():
        print(f"    {cat:<26} {r['correct']:>3} / {r['samples']:<3} ({r['correct_rate']:.2%})")
    print("\n  Decision policy on test set:")
    for label, counts in results["decisions"].items():
        print(f"    {label:<18} {counts}")
