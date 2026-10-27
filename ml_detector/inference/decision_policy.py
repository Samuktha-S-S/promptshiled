"""
PromptShield - Confidence-Aware Decision Policy
Converts ML classification + confidence into security actions (ALLOW / FLAG / BLOCK).

IMPORTANT: The ML classifier performs BINARY classification (benign / prompt_injection).
The FLAG state is produced by this decision layer when confidence is insufficient.

The policy is deliberately separate from the classifiers: it only sees a
predicted label and a confidence, so the same rules apply whichever of the three
algorithms is in production. A BENIGN prediction is never treated as a proof
of safety: below the validated threshold it becomes FLAG, not ALLOW.
"""

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np

ALLOW = "ALLOW"
FLAG = "FLAG"
BLOCK = "BLOCK"

THRESHOLD_KEYS = ("benign_allow_threshold", "injection_block_threshold")


@dataclass
class SecurityDecision:
    """Result of the confidence-aware security decision."""
    classification: str       # "benign" or "prompt_injection"
    confidence: float         # Model confidence in the predicted class
    action: str               # "ALLOW", "FLAG", or "BLOCK"
    reason: str               # Human-readable reason


class DecisionPolicy:
    """
    Confidence-aware decision policy.

    Converts binary ML predictions into three-tier security actions:
    - ALLOW: High-confidence benign -> may be forwarded to the LLM
    - FLAG: Low confidence in either direction -> held for review, no LLM call
    - BLOCK: High-confidence prompt injection -> reject, no LLM call

    Thresholds must come from validation data (see select_thresholds); there
    are no built-in defaults.
    """

    def __init__(self, thresholds: Dict[str, float]):
        missing = [k for k in THRESHOLD_KEYS if k not in thresholds]
        if missing:
            raise ValueError(f"Decision thresholds missing {missing}.")
        for key in THRESHOLD_KEYS:
            value = thresholds[key]
            if not isinstance(value, (int, float)) or not 0.5 <= value <= 1.0:
                raise ValueError(f"Threshold {key}={value!r} must be a number in [0.5, 1.0].")
        self.thresholds = {k: float(thresholds[k]) for k in THRESHOLD_KEYS}

    def decide(self, classification: str, confidence: float) -> SecurityDecision:
        """
        Make a security decision based on classification and confidence.

        Args:
            classification: "benign" or "prompt_injection" from the ML classifier
            confidence: Model confidence in the predicted class (0.0 - 1.0)
        """
        if confidence is None or not isinstance(confidence, (int, float)) or math.isnan(confidence):
            return SecurityDecision(classification, float("nan"), FLAG, "Missing or invalid confidence score.")

        if classification == "benign":
            if confidence >= self.thresholds["benign_allow_threshold"]:
                return SecurityDecision(classification, confidence, ALLOW,
                                        "High-confidence benign classification.")
            return SecurityDecision(classification, confidence, FLAG,
                                    "Low confidence in benign classification. Cannot confidently determine safety.")

        if classification == "prompt_injection":
            if confidence >= self.thresholds["injection_block_threshold"]:
                return SecurityDecision(classification, confidence, BLOCK,
                                        "High-confidence prompt injection detected.")
            return SecurityDecision(classification, confidence, FLAG,
                                    "Possible prompt injection detected but confidence is low.")

        # Unknown classification — never allow
        return SecurityDecision(classification, confidence, FLAG, f"Unknown classification '{classification}'.")

    def get_thresholds(self) -> dict:
        return self.thresholds.copy()


def select_thresholds(p_injection: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Select confidence thresholds using VALIDATION data only.

    Grid-searches both thresholds over 0.55..0.90 and scores each pair:
    a correct ALLOW / BLOCK earns 1.0, a FLAG earns 0.3 (conservative but
    unhelpful), a wrong ALLOW / BLOCK earns 0, and the overall flag rate is
    penalised by 0.1. The first best pair in grid order is kept, i.e. the
    least restrictive thresholds that achieve the best validation score.
    """
    p_injection = np.asarray(p_injection, dtype=float)
    y_true = np.asarray(y_true)
    predicted = (p_injection >= 0.5).astype(int)
    confidence = np.where(predicted == 1, p_injection, 1.0 - p_injection)

    best, best_score = None, -np.inf
    grid = np.round(np.arange(0.55, 0.95, 0.05), 2)
    for benign_t in grid:
        for inject_t in grid:
            allow = (predicted == 0) & (confidence >= benign_t)
            block = (predicted == 1) & (confidence >= inject_t)
            flag = ~(allow | block)
            correct = (allow & (y_true == 0)).sum() + (block & (y_true == 1)).sum() + 0.3 * flag.sum()
            score = correct / len(y_true) - 0.1 * flag.mean()
            if score > best_score:
                best_score = score
                best = {"benign_allow_threshold": float(benign_t), "injection_block_threshold": float(inject_t)}

    return {**best, "validation_objective": float(best_score), "selected_on": "validation"}
