"""FastAPI endpoints, API contract with the frontend, and secret handling."""

import logging
import os
import re

from conftest import (
    ALLOW_PROMPT, BLOCK_PROMPT, FAKE_API_KEY, FLAG_PROMPT, ROOT, FakeDetector, FakeGeminiSDK,
)

FRONTEND_FIELDS = {"classification", "confidence", "action", "response"}  # read by frontend/src/App.jsx


def test_health(make_client):
    body = make_client().get("/api/health").json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["database_connected"] is True
    assert body["llm_configured"] is True


def test_health_degraded_without_database(make_client):
    body = make_client(repository=None).get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["database_connected"] is False


def test_chat_allow(make_client, fake_sdk):
    r = make_client().post("/api/chat", json={"message": ALLOW_PROMPT})
    assert r.status_code == 200
    body = r.json()
    assert FRONTEND_FIELDS <= body.keys()
    assert body["action"] == "ALLOW" and body["llm_status"] == "generated"
    assert body["response"] == fake_sdk.reply
    assert body["classifier"] == "linear_svm"


def test_chat_continues_conversation(make_client, fake_sdk):
    client = make_client()
    cid = client.post("/api/chat", json={"message": ALLOW_PROMPT}).json()["conversation_id"]
    second = client.post("/api/chat", json={"message": "And Germany?", "conversation_id": cid}).json()
    assert second["conversation_id"] == cid
    assert len(fake_sdk.calls[-1]["contents"]) == 3


def test_chat_flag_and_block(make_client, fake_sdk):
    client = make_client()
    flag = client.post("/api/chat", json={"message": FLAG_PROMPT}).json()
    block = client.post("/api/chat", json={"message": BLOCK_PROMPT}).json()
    assert flag["action"] == "FLAG" and flag["llm_status"] == "pending_review"
    assert block["action"] == "BLOCK" and block["llm_status"] == "not_called"
    assert fake_sdk.calls == []


def test_detect_endpoint_never_calls_gemini(make_client, fake_sdk):
    client = make_client()
    body = client.post("/api/detect", json={"message": ALLOW_PROMPT}).json()
    assert body["action"] == "ALLOW"
    assert "response" not in body
    assert fake_sdk.calls == []


def test_validation_errors(make_client):
    client = make_client()
    assert client.post("/api/chat", json={"message": ""}).status_code == 422
    assert client.post("/api/chat", json={"message": "x" * 5000}).status_code == 422
    assert client.post("/api/chat", json={}).status_code == 422
    r = client.post("/api/chat", json={"message": "    "})
    assert r.status_code == 400


def test_detector_failure_returns_503(make_client, fake_sdk):
    r = make_client(detector=FakeDetector(error=RuntimeError("model gone"))).post(
        "/api/chat", json={"message": ALLOW_PROMPT})
    assert r.status_code == 503
    assert "model gone" not in r.text  # internal detail not exposed
    assert fake_sdk.calls == []


def test_database_unavailable_returns_503(make_client, fake_sdk):
    r = make_client(repository=None).post("/api/chat", json={"message": ALLOW_PROMPT})
    assert r.status_code == 503
    assert fake_sdk.calls == []


def test_review_endpoints_require_admin_key(make_client):
    client = make_client()
    event_id = client.post("/api/chat", json={"message": FLAG_PROMPT}).json()["event_id"]
    assert client.get("/api/review/pending").status_code == 401
    assert client.get("/api/review/pending", headers={"X-Admin-Key": "wrong"}).status_code == 401
    assert client.post(f"/api/review/{event_id}", json={"approve": True}).status_code == 401

    disabled = make_client(admin_key=None)
    assert disabled.get("/api/review/pending", headers={"X-Admin-Key": "x"}).status_code == 503


def test_review_flow_via_api(make_client, fake_sdk):
    client = make_client()
    headers = {"X-Admin-Key": "admin-secret"}
    event_id = client.post("/api/chat", json={"message": FLAG_PROMPT}).json()["event_id"]

    pending = client.get("/api/review/pending", headers=headers).json()["items"]
    assert [p["event_id"] for p in pending] == [event_id]
    assert pending[0]["prompt"] == FLAG_PROMPT

    result = client.post(f"/api/review/{event_id}", json={"approve": True}, headers=headers).json()
    assert result["llm_status"] == "generated"
    assert len(fake_sdk.calls) == 1


def test_blocked_request_cannot_be_released_via_review(make_client, fake_sdk):
    client = make_client()
    event_id = client.post("/api/chat", json={"message": BLOCK_PROMPT}).json()["event_id"]
    r = client.post(f"/api/review/{event_id}", json={"approve": True}, headers={"X-Admin-Key": "admin-secret"})
    assert r.status_code == 409
    assert fake_sdk.calls == []


def test_api_key_never_exposed(make_client, caplog):
    leaky = FakeGeminiSDK(error=RuntimeError(f"401 for key {FAKE_API_KEY}"))
    client = make_client(sdk=leaky)
    with caplog.at_level(logging.DEBUG):
        bodies = [
            client.get("/api/health").text,
            client.post("/api/chat", json={"message": ALLOW_PROMPT}).text,
            client.post("/api/detect", json={"message": ALLOW_PROMPT}).text,
        ]
    assert len(leaky.calls) == 1
    for text in bodies + [caplog.text]:
        assert FAKE_API_KEY not in text


def test_no_unlisted_endpoint_reaches_gemini(make_client):
    """Every route that can trigger generation is known and gated."""
    client = make_client()
    api_paths = {p for p in client.app.openapi()["paths"] if p.startswith("/api")}
    assert api_paths == {"/api/chat", "/api/detect", "/api/health", "/api/review/pending", "/api/review/{event_id}"}


# --------------------------------------------------------------------------- frontend compatibility

def test_frontend_calls_existing_endpoints():
    api_js = open(os.path.join(ROOT, "frontend", "src", "services", "api.js"), encoding="utf-8").read()
    assert "${API_BASE}/chat" in api_js and "${API_BASE}/health" in api_js
    assert "const API_BASE = '/api'" in api_js
    body_keys = set(re.findall(r"JSON\.stringify\(\{([^}]*)\}\)", api_js)[0].replace(" ", "").split(","))
    assert {k.split(":")[0] for k in body_keys} <= {"message", "conversation_id"}


def test_frontend_proxy_targets_backend_port():
    vite = open(os.path.join(ROOT, "frontend", "vite.config.js"), encoding="utf-8").read()
    assert "http://localhost:8000" in vite


def test_frontend_has_no_gemini_access():
    for dirpath, _, files in os.walk(os.path.join(ROOT, "frontend", "src")):
        for name in files:
            text = open(os.path.join(dirpath, name), encoding="utf-8").read().lower()
            assert "gemini_api_key" not in text
            assert "generativelanguage.googleapis.com" not in text
            assert "@google/genai" not in text


def test_env_example_has_no_real_secrets():
    text = open(os.path.join(ROOT, ".env.example"), encoding="utf-8").read()
    values = dict(line.split("=", 1) for line in text.splitlines() if "=" in line and not line.startswith("#"))
    assert values["GEMINI_API_KEY"].strip() == ""
    assert not re.search(r"AIza[0-9A-Za-z_\-]{20,}", text)
