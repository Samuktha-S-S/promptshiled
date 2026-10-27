"""
PromptShield - Prompt Injection Detector
The single inference entry point used by the backend:

    prompt -> validate -> Sentence-BERT embedding -> configured classifier -> DetectionResult

Only ONE classifier (the production one) runs per prompt. Which one is decided
by, in order:
  1. the `algorithm` argument / PROMPTSHIELD_CLASSIFIER environment variable
  2. ml_detector/models/production_model.json (written by evaluation/compare.py)

Any failure raises DetectorUnavailableError or DetectorInferenceError; the
detector never falls back to "benign". This module has no dependency on the
LLM package.
"""

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional

from ml_detector.algorithms.common import ArtifactError, load_artifact
from ml_detector.algorithms.linear_svm import predict as svm_predict
from ml_detector.algorithms.logistic_regression import predict as lr_predict
from ml_detector.algorithms.random_forest import predict as rf_predict
from ml_detector.config import LABELS, MODELS_DIR, PRODUCTION_CONFIG_FILE
from ml_detector.embeddings import EmbeddingError, get_embedder
from ml_detector.inference.decision_policy import DecisionPolicy
from ml_detector.preprocessing import validate_prompt

PREDICTORS = {
    "logistic_regression": lr_predict.predict,
    "linear_svm": svm_predict.predict,
    "random_forest": rf_predict.predict,
}


class DetectorError(RuntimeError):
    """Base class: the detector could not produce a trustworthy prediction."""


class DetectorUnavailableError(DetectorError):
    """The model or its artifacts could not be loaded."""


class DetectorInferenceError(DetectorError):
    """The model was loaded but failed on this prompt."""


@dataclass
class DetectionResult:
    """Prediction for one prompt, independent of which classifier produced it."""
    classification: str              # "benign" or "prompt_injection"
    confidence: float                # score of the predicted class (input to the decision policy)
    p_injection: float               # injection score in [0, 1]; see score_type
    score_type: str                  # e.g. "platt_calibrated_probability"
    decision_score: Optional[float]  # raw SVM margin (not a probability); None for other models
    algorithm: str
    embedding_ms: float
    classification_ms: float
    total_ms: float


def resolve_production_algorithm(models_dir: str = MODELS_DIR) -> str:
    """Return the configured production algorithm name."""
    algorithm = os.getenv("PROMPTSHIELD_CLASSIFIER", "").strip()
    if not algorithm:
        path = os.path.join(models_dir, PRODUCTION_CONFIG_FILE)
        if not os.path.exists(path):
            raise DetectorUnavailableError(
                f"No production model configured ({path} missing and PROMPTSHIELD_CLASSIFIER unset). "
                "Run 'python -m ml_detector.evaluation.compare' after training."
            )
        with open(path) as f:
            algorithm = json.load(f).get("algorithm", "")
    if algorithm not in PREDICTORS:
        raise DetectorUnavailableError(
            f"Unknown classifier '{algorithm}'. Expected one of {sorted(PREDICTORS)}."
        )
    return algorithm


class PromptInjectionDetector:
    """Loads the embedding model and one trained classifier and serves predictions."""

    def __init__(self, algorithm: Optional[str] = None, models_dir: str = MODELS_DIR, embedder=None):
        self.models_dir = models_dir
        self.algorithm = algorithm
        self.embedder = embedder
        self.model = None
        self.metadata: dict = {}
        self.policy: Optional[DecisionPolicy] = None
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def model_version(self) -> str:
        if not self.is_loaded:
            return "not-loaded"
        return f"{self.algorithm}@{self.metadata.get('trained_at', 'unknown')}"

    def load(self) -> None:
        with self._lock:
            if self.model is not None:
                return
            try:
                self.algorithm = self.algorithm or resolve_production_algorithm(self.models_dir)
                if self.algorithm not in PREDICTORS:
                    raise DetectorUnavailableError(f"Unknown classifier '{self.algorithm}'.")
                self.embedder = self.embedder or get_embedder()
                self.embedder.load()
                model, metadata = load_artifact(self.algorithm, self.models_dir, self.embedder.info())
                self.policy = DecisionPolicy(metadata["thresholds"])
            except DetectorUnavailableError:
                raise
            except (ArtifactError, EmbeddingError, KeyError, ValueError) as e:
                raise DetectorUnavailableError(f"Detector could not be loaded: {e}") from e
            self.metadata = metadata
            self.model = model

    def detect(self, text: str) -> DetectionResult:
        is_valid, error = validate_prompt(text)
        if not is_valid:
            raise ValueError(error)
        if not self.is_loaded:
            self.load()

        total_start = time.perf_counter()
        try:
            start = time.perf_counter()
            embedding = self.embedder.encode(text)
            embedding_ms = (time.perf_counter() - start) * 1000

            start = time.perf_counter()
            out = PREDICTORS[self.algorithm](self.model, embedding)
            classification_ms = (time.perf_counter() - start) * 1000
        except Exception as e:
            raise DetectorInferenceError(f"Inference failed: {type(e).__name__}") from e

        p_injection = float(out.p_injection[0])
        label = LABELS[int(out.labels[0])]
        return DetectionResult(
            classification=label,
            confidence=p_injection if label == "prompt_injection" else 1.0 - p_injection,
            p_injection=p_injection,
            score_type=out.score_type,
            decision_score=None if out.decision_scores is None else float(out.decision_scores[0]),
            algorithm=self.algorithm,
            embedding_ms=embedding_ms,
            classification_ms=classification_ms,
            total_ms=(time.perf_counter() - total_start) * 1000,
        )
