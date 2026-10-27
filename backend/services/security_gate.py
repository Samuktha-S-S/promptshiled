"""
PromptShield - Security Gate
Enforces the ML -> decision policy -> LLM workflow on the backend:

    prompt -> Sentence-BERT -> classifier -> decision policy
        ALLOW -> Gemini (authorised)            -> reply stored and returned
        FLAG  -> held for human review          -> Gemini only after an admin approves
        BLOCK -> rejected                       -> Gemini never called
        detector failure -> controlled 503      -> Gemini never called, never defaults to ALLOW

Every request is recorded in PostgreSQL before anything is returned. If the
database is unavailable the request is refused, because security events could
not be recorded.
"""

import functools
from dataclasses import dataclass
from typing import List, Optional

from sqlalchemy.exc import SQLAlchemyError

from backend.core.security import authorize_generation, authorize_reviewed_generation
from backend.database import repository as repo_status
from backend.database.connection import DB_STATUS_MESSAGES, DatabaseUnavailableError, classify_db_error
from llm.chat_service import ChatService
from llm.gemini_client import (
    GeminiAuthenticationError, GeminiConfigurationError, GeminiEmptyResponseError, GeminiError,
    GeminiOverloadedError, GeminiRateLimitError, GeminiTimeoutError,
)
from ml_detector.inference.decision_policy import ALLOW, BLOCK, FLAG, SecurityDecision
from ml_detector.inference.detector import DetectionResult, DetectorError
from ml_detector.preprocessing import validate_prompt

FLAG_MESSAGE = (
    "This request could not be confidently classified as safe. It has been held for "
    "review and was not sent to the AI model. You can also rephrase your request and try again."
)
BLOCK_MESSAGE = (
    "This message was detected as a potential prompt injection "
    "attempt and was not forwarded to the AI model."
)
DETECTOR_FAILURE_MESSAGE = (
    "The security classifier is unavailable, so this request was not processed. Please try again later."
)


# Stable codes the frontend uses to explain why no AI response was produced.
LLM_ERROR_CODES = {
    GeminiConfigurationError: "llm_not_configured",
    GeminiAuthenticationError: "llm_auth_failed",
    GeminiRateLimitError: "llm_rate_limited",
    GeminiTimeoutError: "llm_timeout",
    GeminiEmptyResponseError: "llm_empty_response",
    GeminiOverloadedError: "llm_overloaded",
}

DATABASE_FAILURE_MESSAGE = "Your message was not processed because the database is unavailable."


class GateError(Exception):
    """
    A controlled failure that the API turns into an HTTP error response:
    {"detail": <safe message>, "code": <stable error code>, "category": <optional sub-category>}
    """

    def __init__(self, status_code: int, detail: str, code: str = "error", category: Optional[str] = None):
        super().__init__(detail)
        self.status_code = status_code
        self.detail = detail
        self.code = code
        self.category = category


def _database_error(exc: BaseException) -> GateError:
    category = classify_db_error(exc)
    return GateError(503, f"{DATABASE_FAILURE_MESSAGE} {DB_STATUS_MESSAGES.get(category, '')}".strip(),
                     code="database_unavailable", category=category)


def database_guard(method):
    """Turn database failures raised anywhere in a gate operation into a 503 GateError.
    The request is reported as failed, never as successful."""

    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except GateError:
            raise
        except (DatabaseUnavailableError, SQLAlchemyError) as e:
            self.audit.log_error("database_error", f"{classify_db_error(e)}: {type(e).__name__}")
            raise _database_error(e) from e

    return wrapper


@dataclass
class Detection:
    result: DetectionResult
    decision: SecurityDecision
    event_id: int


class SecurityGate:
    def __init__(self, detector, chat_service: ChatService, repository, audit, history_turns: int = 10,
                 max_prompt_length: int = 4096):
        self.detector = detector
        self.chat_service = chat_service
        self.repository = repository
        self.audit = audit
        self.history_turns = history_turns
        self.max_prompt_length = max_prompt_length

    # ------------------------------------------------------------------ helpers

    def _require_repository(self):
        if self.repository is None:
            raise GateError(503, f"{DATABASE_FAILURE_MESSAGE} {DB_STATUS_MESSAGES['not_configured']}",
                            code="database_unavailable", category="not_configured")
        return self.repository

    def _validate(self, message: str) -> None:
        is_valid, error = validate_prompt(message, self.max_prompt_length)
        if not is_valid:
            raise GateError(400, error, code="invalid_input")

    def _detect(self, message: str, source: str, conversation_id: Optional[str]) -> Detection:
        repo = self._require_repository()
        try:
            if self.detector is None:
                raise DetectorError("Detector not initialised.")
            result = self.detector.detect(message)
            decision = self.detector.policy.decide(result.classification, result.confidence)
        except Exception as e:
            # Fail closed: no decision, no LLM call, controlled error.
            error_type = type(e).__name__
            self.audit.log_error("detector_error", f"{error_type}: {e}")
            event_id = repo.create_event(
                conversation_id=conversation_id, source=source, prompt_length=len(message),
                response_status=repo_status.DETECTOR_ERROR, error_type=error_type, llm_called=False,
            )
            raise GateError(503, DETECTOR_FAILURE_MESSAGE, code="detector_unavailable") from e

        self.audit.log_classification(
            classification=decision.classification, confidence=decision.confidence, action=decision.action,
            model_version=self.detector.model_version, latency_ms=result.total_ms, prompt_length=len(message),
        )
        event_id = repo.create_event(
            conversation_id=conversation_id, source=source, prompt_length=len(message),
            algorithm=result.algorithm, model_version=self.detector.model_version,
            classification=decision.classification, confidence=decision.confidence,
            p_injection=result.p_injection, score_type=result.score_type, latency_ms=result.total_ms,
            decision=decision.action, reason=decision.reason, llm_called=False,
            response_status=repo_status.IN_PROGRESS,
        )
        return Detection(result=result, decision=decision, event_id=event_id)

    def _generate(self, authorization, prompt: str, conversation_id: str, before_message_id: int, event_id: int):
        """Call Gemini through the chat service. Returns (llm_status, response_text, llm_error_code)."""
        repo = self.repository
        if not self.chat_service.is_configured:
            repo.update_event(event_id, response_status=repo_status.LLM_ERROR,
                              error_type=GeminiConfigurationError.__name__, llm_called=False)
            self.audit.log_error("llm_not_configured", "GEMINI_API_KEY is not set.")
            return "error", GeminiConfigurationError.user_message, LLM_ERROR_CODES[GeminiConfigurationError]

        history = repo.allowed_history(conversation_id, before_message_id, self.history_turns)
        try:
            reply = self.chat_service.generate_reply(authorization, prompt, history)
        except GeminiError as e:
            repo.update_event(event_id, response_status=repo_status.LLM_ERROR,
                              error_type=type(e).__name__, llm_called=True)
            self.audit.log_error("llm_error", f"{type(e).__name__}: {e}")
            return "error", e.user_message, LLM_ERROR_CODES.get(type(e), "llm_unavailable")

        repo.add_message(conversation_id, "assistant", reply, event_id)
        repo.update_event(event_id, response_status=repo_status.GENERATED, llm_called=True)
        return "generated", reply, None

    # ------------------------------------------------------------------ chat

    @database_guard
    def process_chat(self, message: str, conversation_id: Optional[str]) -> dict:
        self._validate(message)
        repo = self._require_repository()

        conversation_id = repo.ensure_conversation(conversation_id)
        message_id = repo.add_message(conversation_id, "user", message)
        detection = self._detect(message, "chat", conversation_id)
        repo.link_message(message_id, detection.event_id)
        decision, event_id = detection.decision, detection.event_id

        if decision.action == ALLOW:
            authorization = authorize_generation(decision, event_id)
            llm_status, response, llm_error = self._generate(
                authorization, message, conversation_id, message_id, event_id)
        elif decision.action == FLAG:
            repo.update_event(event_id, response_status=repo_status.PENDING_REVIEW, review_status="pending")
            llm_status, response, llm_error = "pending_review", FLAG_MESSAGE, None
        else:  # BLOCK, or anything unexpected: never reaches the LLM
            repo.update_event(event_id, response_status=repo_status.BLOCKED)
            llm_status, response, llm_error = "not_called", BLOCK_MESSAGE, None

        return {
            "classification": decision.classification,
            "confidence": round(decision.confidence, 4),
            "action": decision.action,
            "response": response,
            "latency_ms": round(detection.result.total_ms, 2),
            "conversation_id": conversation_id,
            "event_id": event_id,
            "reason": decision.reason,
            "p_injection": round(detection.result.p_injection, 4),
            "score_type": detection.result.score_type,
            "classifier": detection.result.algorithm,
            "llm_status": llm_status,
            "llm_error_code": llm_error,
        }

    # ------------------------------------------------------------------ detection only

    @database_guard
    def process_detect(self, message: str) -> dict:
        self._validate(message)
        detection = self._detect(message, "detect", None)
        self.repository.update_event(detection.event_id, response_status=repo_status.DETECTION_ONLY)
        r, d = detection.result, detection.decision
        return {
            "classification": d.classification,
            "confidence": round(d.confidence, 4),
            "action": d.action,
            "reason": d.reason,
            "p_injection": round(r.p_injection, 4),
            "score_type": r.score_type,
            "decision_score": None if r.decision_score is None else round(r.decision_score, 4),
            "classifier": r.algorithm,
            "latency_ms": round(r.total_ms, 2),
            "event_id": detection.event_id,
        }

    # ------------------------------------------------------------------ FLAG review

    @database_guard
    def pending_reviews(self) -> List[dict]:
        return self._require_repository().pending_reviews()

    @database_guard
    def review(self, event_id: int, approve: bool) -> dict:
        repo = self._require_repository()
        event = repo.get_event(event_id)
        if event is None:
            raise GateError(404, "Detection event not found.", code="not_found")
        if event["decision"] != FLAG or event["source"] != "chat":
            raise GateError(409, f"Only flagged chat requests can be reviewed (decision={event['decision']}).",
                            code="conflict")
        if approve and not self.chat_service.is_configured:
            raise GateError(503, GeminiConfigurationError.user_message, code="llm_unavailable",
                            category="not_configured")

        if not repo.claim_review(event_id, "approved" if approve else "rejected"):
            raise GateError(409, "This request has already been reviewed.", code="conflict")

        if not approve:
            repo.update_event(event_id, response_status=repo_status.REJECTED_BY_REVIEW)
            return {"event_id": event_id, "review_status": "rejected", "llm_status": "not_called", "response": None}

        authorization = authorize_reviewed_generation(event["decision"], event_id)
        llm_status, response, _ = self._generate(
            authorization, event["prompt"], event["conversation_id"], event["message_id"], event_id
        )
        return {"event_id": event_id, "review_status": "approved", "llm_status": llm_status, "response": response}
