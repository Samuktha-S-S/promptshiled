"""
PromptShield - Chat API Schemas
ChatResponse keeps every field of the original API (classification,
confidence, action, response, latency_ms) and adds new ones, so existing
callers keep working.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

LLMStatus = Literal["generated", "not_called", "pending_review", "error"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)
    conversation_id: Optional[str] = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    classification: str
    confidence: float
    action: Literal["ALLOW", "FLAG", "BLOCK"]
    response: str
    latency_ms: Optional[float] = None
    # Added fields
    conversation_id: str
    event_id: int
    reason: str
    p_injection: float
    score_type: str
    classifier: str
    llm_status: LLMStatus
    llm_error_code: Optional[str] = None  # e.g. llm_not_configured, llm_rate_limited, llm_timeout


class ReviewItem(BaseModel):
    event_id: int
    conversation_id: Optional[str]
    prompt: Optional[str]
    classification: Optional[str]
    confidence: Optional[float]
    p_injection: Optional[float]
    reason: Optional[str]
    created_at: datetime


class ReviewList(BaseModel):
    items: List[ReviewItem]


class ReviewDecisionRequest(BaseModel):
    approve: bool


class ReviewDecisionResponse(BaseModel):
    event_id: int
    review_status: Literal["approved", "rejected"]
    llm_status: LLMStatus
    response: Optional[str] = None
