"""
PromptShield - Backend Security Helpers

* authorize_generation(): the ONLY place that turns a decision-policy result
  into permission to call the LLM. Only ALLOW is authorised here; FLAG needs a
  human reviewer (authorize_reviewed_generation) and BLOCK is never authorised.
* require_admin(): protects the FLAG review endpoints with ADMIN_API_KEY.
"""

import secrets
from typing import Optional

from fastapi import Header, HTTPException, Request

from llm.chat_service import GenerationAuthorization
from ml_detector.inference.decision_policy import ALLOW, FLAG, SecurityDecision


def authorize_generation(decision: SecurityDecision, event_id: Optional[int] = None) -> Optional[GenerationAuthorization]:
    """Return an authorisation for ALLOW decisions, otherwise None."""
    if decision is not None and decision.action == ALLOW:
        return GenerationAuthorization(action=ALLOW, event_id=event_id)
    return None


def authorize_reviewed_generation(event_decision: str, event_id: int) -> GenerationAuthorization:
    """Authorisation for a FLAG decision that an administrator has approved."""
    if event_decision != FLAG:
        raise PermissionError(f"Only FLAG decisions can be approved by review (got {event_decision}).")
    return GenerationAuthorization(action=FLAG, event_id=event_id, human_approved=True)


def require_admin(request: Request, x_admin_key: Optional[str] = Header(default=None)) -> None:
    """FastAPI dependency: the X-Admin-Key header must match ADMIN_API_KEY."""
    expected = request.app.state.settings.admin_api_key
    if not expected:
        raise HTTPException(status_code=503, detail="Review workflow is disabled (ADMIN_API_KEY not set).")
    if not x_admin_key or not secrets.compare_digest(x_admin_key, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing admin key.")
