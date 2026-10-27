"""
PromptShield - Evaluation Metrics
One implementation of every metric, used by all three algorithms' evaluate.py
and by the comparison, so the numbers are computed identically.

Positive class = prompt_injection (label 1).
"""

import os
from collections import Counter
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from ml_detector.config import LABELS


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Accuracy, precision, recall, F1, FPR, FNR and the confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())
    return {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positive_rate": fp / (fp + tn) if (fp + tn) else 0.0,
        "false_negative_rate": fn / (fn + tp) if (fn + tp) else 0.0,
        "confusion_matrix": cm.tolist(),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }


def per_category_metrics(frame: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, dict]:
    """
    Per attack_type: number of samples and the fraction classified correctly.
    For attack categories this is the detection rate; for 'normal' it is the
    fraction of benign prompts correctly left as benign.
    """
    results = {}
    for category in sorted(frame["attack_type"].unique()):
        mask = frame["attack_type"].values == category
        results[category] = {
            "samples": int(mask.sum()),
            "correct": int((y_true[mask] == y_pred[mask]).sum()),
            "correct_rate": float((y_true[mask] == y_pred[mask]).mean()),
        }
    return results


def decision_breakdown(actions: Sequence[str], y_true: np.ndarray) -> dict:
    """How the ALLOW / FLAG / BLOCK policy treats benign vs. injection prompts."""
    breakdown = {}
    for label_idx, label in enumerate(LABELS):
        mask = y_true == label_idx
        counts = Counter(a for a, m in zip(actions, mask) if m)
        breakdown[label] = {action: int(counts.get(action, 0)) for action in ("ALLOW", "FLAG", "BLOCK")}
    return breakdown


def plot_confusion_matrix(cm, title: str, filepath: str) -> Optional[str]:
    """Save a confusion matrix image. Returns None if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
    ax.set_xticks([0, 1], labels=LABELS)
    ax.set_yticks([0, 1], labels=LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath
