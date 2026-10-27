"""
PromptShield - Detection API Schemas
/api/detect runs the classifier and decision policy only; it never calls the LLM.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class DetectRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)


class DetectResponse(BaseModel):
    classification: str
    confidence: float
    action: Literal["ALLOW", "FLAG", "BLOCK"]
    reason: str
    p_injection: float
    score_type: str
    decision_score: Optional[float] = Field(
        default=None, description="Raw Linear SVM margin; not a probability. Null for other classifiers."
    )
    classifier: str
    latency_ms: float
    event_id: int
