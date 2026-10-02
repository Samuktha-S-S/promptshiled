"""
PromptShield - Logging Module
Structured logging for security events without exposing sensitive prompt content.
"""

import logging
import json
from datetime import datetime, timezone
from typing import Optional


class PromptShieldLogger:
    """Structured logger for PromptShield security events."""

    def __init__(self, name: str = "promptshield"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_classification(
        self,
        classification: str,
        confidence: float,
        action: str,
        model_version: str,
        latency_ms: float,
        prompt_length: Optional[int] = None
    ):
        """Log a classification event without storing the prompt content."""
        event = {
            "event": "classification",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "classification": classification,
            "confidence": round(confidence, 4),
            "action": action,
            "model_version": model_version,
            "latency_ms": round(latency_ms, 2),
        }
        if prompt_length is not None:
            event["prompt_length"] = prompt_length
        self.logger.info(json.dumps(event))

    def log_startup(self, model_version: str, thresholds: dict):
        """Log system startup."""
        event = {
            "event": "startup",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model_version": model_version,
            "thresholds": thresholds,
        }
        self.logger.info(json.dumps(event))

    def log_error(self, error_type: str, message: str):
        """Log an error event."""
        event = {
            "event": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error_type": error_type,
            "message": message,
        }
        self.logger.error(json.dumps(event))


# Singleton instance
logger = PromptShieldLogger()
