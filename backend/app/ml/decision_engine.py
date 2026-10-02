"""
PromptShield - Confidence-Aware Decision Engine
Converts ML classification + confidence into security actions (ALLOW / FLAG / BLOCK).

IMPORTANT: The ML classifier performs BINARY classification (benign / prompt_injection).
The SUSPICIOUS/FLAG state is produced by this decision layer when confidence is insufficient.
"""

import json
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class SecurityDecision:
    """Result of the confidence-aware security decision."""
    classification: str       # "benign" or "prompt_injection"
    confidence: float         # Model confidence in the predicted class
    action: str               # "ALLOW", "FLAG", or "BLOCK"
    reason: str               # Human-readable reason


class DecisionEngine:
    """
    Confidence-aware decision engine.
    
    Converts binary ML predictions into three-tier security actions:
    - ALLOW: High-confidence benign → forward to LLM
    - FLAG: Low confidence in either direction → ask user to rephrase
    - BLOCK: High-confidence prompt injection → reject, no LLM call
    
    Thresholds are determined from the validation set during training,
    NOT hardcoded arbitrarily.
    """
    
    DEFAULT_THRESHOLDS = {
        "benign_allow_threshold": 0.75,     # Min confidence to ALLOW benign
        "injection_block_threshold": 0.70,  # Min confidence to BLOCK injection
    }
    
    def __init__(self, thresholds: Optional[dict] = None):
        if thresholds is not None:
            self.thresholds = thresholds
        else:
            self.thresholds = self.DEFAULT_THRESHOLDS.copy()
    
    def decide(self, classification: str, confidence: float) -> SecurityDecision:
        """
        Make a security decision based on classification and confidence.
        
        Args:
            classification: "benign" or "prompt_injection" from the ML classifier
            confidence: Probability confidence in the predicted class (0.0 - 1.0)
        
        Returns:
            SecurityDecision with action ALLOW, FLAG, or BLOCK
        """
        if classification == "benign":
            if confidence >= self.thresholds["benign_allow_threshold"]:
                return SecurityDecision(
                    classification=classification,
                    confidence=confidence,
                    action="ALLOW",
                    reason="High-confidence benign classification."
                )
            else:
                return SecurityDecision(
                    classification=classification,
                    confidence=confidence,
                    action="FLAG",
                    reason="Low confidence in benign classification. Cannot confidently determine safety."
                )
        
        elif classification == "prompt_injection":
            if confidence >= self.thresholds["injection_block_threshold"]:
                return SecurityDecision(
                    classification=classification,
                    confidence=confidence,
                    action="BLOCK",
                    reason="High-confidence prompt injection detected."
                )
            else:
                return SecurityDecision(
                    classification=classification,
                    confidence=confidence,
                    action="FLAG",
                    reason="Possible prompt injection detected but confidence is low."
                )
        
        else:
            # Unknown classification — always flag
            return SecurityDecision(
                classification=classification,
                confidence=confidence,
                action="FLAG",
                reason=f"Unknown classification '{classification}'."
            )
    
    def save(self, path: str):
        """Save thresholds to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.thresholds, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> "DecisionEngine":
        """Load thresholds from JSON."""
        with open(path, 'r') as f:
            thresholds = json.load(f)
        return cls(thresholds=thresholds)
    
    def get_thresholds(self) -> dict:
        """Return current thresholds."""
        return self.thresholds.copy()
