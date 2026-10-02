"""
PromptShield - Prediction Module
Runs the full ML prediction pipeline for a single prompt.
"""

import time
import numpy as np
from typing import Optional
from dataclasses import dataclass

from app.ml.preprocess import preprocess_text
from app.ml.decision_engine import SecurityDecision, DecisionEngine


@dataclass
class PredictionResult:
    """Complete prediction result including latency breakdown."""
    classification: str
    confidence: float
    action: str
    reason: str
    preprocess_ms: float
    embedding_ms: float
    classification_ms: float
    total_ms: float


class PromptClassifier:
    """
    Full prediction pipeline:
    Preprocess → Embed → Classify → Decision
    """
    
    def __init__(self, model, embedder, decision_engine: DecisionEngine, label_map: dict):
        self.model = model
        self.embedder = embedder
        self.decision_engine = decision_engine
        self.label_map = label_map  # {0: "benign", 1: "prompt_injection"}
        self.reverse_label_map = {v: k for k, v in label_map.items()}
    
    def predict(self, text: str) -> PredictionResult:
        """
        Run the complete prediction pipeline on a single prompt.
        
        Returns PredictionResult with classification, confidence, action, and latency.
        """
        total_start = time.time()
        
        # Step 1: Preprocess
        t0 = time.time()
        processed = preprocess_text(text)
        preprocess_ms = (time.time() - t0) * 1000
        
        # Step 2: Embed
        embedding, embedding_ms = self.embedder.embed(processed)
        
        # Step 3: Classify
        t0 = time.time()
        embedding_2d = embedding.reshape(1, -1)
        
        # Get probabilities
        probas = self.model.predict_proba(embedding_2d)[0]
        predicted_idx = np.argmax(probas)
        confidence = float(probas[predicted_idx])
        classification = self.label_map[predicted_idx]
        
        classification_ms = (time.time() - t0) * 1000
        
        # Step 4: Security decision
        decision = self.decision_engine.decide(classification, confidence)
        
        total_ms = (time.time() - total_start) * 1000
        
        return PredictionResult(
            classification=decision.classification,
            confidence=decision.confidence,
            action=decision.action,
            reason=decision.reason,
            preprocess_ms=preprocess_ms,
            embedding_ms=embedding_ms,
            classification_ms=classification_ms,
            total_ms=total_ms,
        )
