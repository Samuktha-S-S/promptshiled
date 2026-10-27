"""The backend security gate between the ML decision and Gemini."""

import pytest

from backend.core.security import authorize_generation, authorize_reviewed_generation
from backend.database import repository as status
from backend.services.security_gate import GateError
from conftest import ALLOW_PROMPT, BLOCK_PROMPT, DEFAULT_SCORES, FLAG_PROMPT, FakeDetector, FakeGeminiSDK
from ml_detector.inference.decision_policy import SecurityDecision
from ml_detector.inference.detector import DetectorInferenceError, DetectorUnavailableError


def test_allow_calls_gemini_and_stores_reply(make_gate, fake_sdk, fake_repo):
    out = make_gate().process_chat(ALLOW_PROMPT, None)
    assert out["action"] == "ALLOW"
    assert out["llm_status"] == "generated"
    assert out["response"] == fake_sdk.reply
    assert len(fake_sdk.calls) == 1
    event = fake_repo.events[out["event_id"]]
    assert event["response_status"] == status.GENERATED and event["llm_called"] is True
    assert [m.role for m in fake_repo.messages] == ["user", "assistant"]


def test_flag_does_not_call_gemini(make_gate, fake_sdk, fake_repo):
    out = make_gate().process_chat(FLAG_PROMPT, None)
    assert out["action"] == "FLAG"
    assert out["llm_status"] == "pending_review"
    assert fake_sdk.calls == []
    event = fake_repo.events[out["event_id"]]
    assert event["review_status"] == "pending" and event["llm_called"] is False


def test_block_never_calls_gemini(make_gate, fake_sdk, fake_repo):
    out = make_gate().process_chat(BLOCK_PROMPT, None)
    assert out["action"] == "BLOCK"
    assert out["llm_status"] == "not_called"
    assert fake_sdk.calls == []
    assert fake_repo.events[out["event_id"]]["response_status"] == status.BLOCKED
    assert [m.role for m in fake_repo.messages] == ["user"]


@pytest.mark.parametrize("error", [DetectorUnavailableError("no model"), DetectorInferenceError("boom"),
                                   RuntimeError("unexpected")])
def test_detector_failure_never_calls_gemini(make_gate, fake_sdk, fake_repo, error):
    gate = make_gate(detector=FakeDetector(error=error))
    with pytest.raises(GateError) as info:
        gate.process_chat(ALLOW_PROMPT, None)
    assert info.value.status_code == 503
    assert fake_sdk.calls == []
    events = list(fake_repo.events.values())
    assert events[-1]["response_status"] == status.DETECTOR_ERROR
    assert events[-1]["decision"] is None  # no default ALLOW


def test_unloaded_detector_fails_closed(make_gate, fake_sdk):
    with pytest.raises(GateError):
        make_gate(detector=FakeDetector(loaded=False)).process_chat(ALLOW_PROMPT, None)
    assert fake_sdk.calls == []


def test_database_unavailable_fails_closed(make_gate, fake_sdk):
    with pytest.raises(GateError) as info:
        make_gate(repository=None).process_chat(ALLOW_PROMPT, None)
    assert info.value.status_code == 503
    assert fake_sdk.calls == []


def test_llm_failure_does_not_invent_reply(make_gate, fake_repo):
    sdk = FakeGeminiSDK(error=TimeoutError())
    out = make_gate(sdk=sdk).process_chat(ALLOW_PROMPT, None)
    assert out["action"] == "ALLOW"
    assert out["llm_status"] == "error"
    assert "took too long" in out["response"]
    assert [m.role for m in fake_repo.messages] == ["user"]  # nothing stored as an assistant reply
    assert fake_repo.events[out["event_id"]]["error_type"] == "GeminiTimeoutError"


def test_llm_not_configured(make_gate, fake_sdk):
    out = make_gate(sdk=None, api_key=None).process_chat(ALLOW_PROMPT, None)
    assert out["llm_status"] == "error"
    assert "not configured" in out["response"]


def test_history_contains_only_answered_allowed_turns(make_gate, fake_sdk):
    gate = make_gate()
    first = gate.process_chat(ALLOW_PROMPT, None)
    cid = first["conversation_id"]
    gate.process_chat(BLOCK_PROMPT, cid)
    gate.process_chat(FLAG_PROMPT, cid)
    gate.process_chat("Second safe question", cid)

    contents = fake_sdk.calls[-1]["contents"]
    texts = [c.parts[0].text for c in contents]
    assert texts == [ALLOW_PROMPT, fake_sdk.reply, "Second safe question"]
    assert BLOCK_PROMPT not in texts and FLAG_PROMPT not in texts


def test_authorize_generation_only_for_allow():
    assert authorize_generation(SecurityDecision("benign", 0.9, "ALLOW", "")) is not None
    assert authorize_generation(SecurityDecision("benign", 0.5, "FLAG", "")) is None
    assert authorize_generation(SecurityDecision("prompt_injection", 0.9, "BLOCK", "")) is None
    assert authorize_generation(None) is None
    with pytest.raises(PermissionError):
        authorize_reviewed_generation("BLOCK", 1)


# --------------------------------------------------------------------------- review workflow

def test_review_approval_calls_gemini_once(make_gate, fake_sdk, fake_repo):
    gate = make_gate()
    flagged = gate.process_chat(FLAG_PROMPT, None)
    assert [i["id"] for i in gate.pending_reviews()] == [flagged["event_id"]]

    result = gate.review(flagged["event_id"], approve=True)
    assert result["review_status"] == "approved" and result["llm_status"] == "generated"
    assert len(fake_sdk.calls) == 1
    assert fake_sdk.calls[0]["contents"][-1].parts[0].text == FLAG_PROMPT

    with pytest.raises(GateError) as info:  # cannot be approved twice
        gate.review(flagged["event_id"], approve=True)
    assert info.value.status_code == 409
    assert len(fake_sdk.calls) == 1


def test_review_rejection(make_gate, fake_sdk, fake_repo):
    gate = make_gate()
    flagged = gate.process_chat(FLAG_PROMPT, None)
    result = gate.review(flagged["event_id"], approve=False)
    assert result["review_status"] == "rejected"
    assert fake_sdk.calls == []
    assert fake_repo.events[flagged["event_id"]]["response_status"] == status.REJECTED_BY_REVIEW


def test_blocked_request_cannot_be_approved(make_gate, fake_sdk):
    gate = make_gate()
    blocked = gate.process_chat(BLOCK_PROMPT, None)
    with pytest.raises(GateError) as info:
        gate.review(blocked["event_id"], approve=True)
    assert info.value.status_code == 409
    assert fake_sdk.calls == []


def test_detect_never_calls_gemini(make_gate, fake_sdk):
    gate = make_gate()
    for prompt in DEFAULT_SCORES:
        gate.process_detect(prompt)
    assert fake_sdk.calls == []
