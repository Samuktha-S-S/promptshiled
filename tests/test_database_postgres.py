"""
PostgreSQL integration tests.

Run with a database the tests may write to, e.g.
    TEST_DATABASE_URL=postgresql://user:password@localhost:5432/promptshield_test pytest -m postgres

Isolation: everything happens inside a freshly created, randomly named schema
which is dropped afterwards, so existing tables are never touched.
"""

import os
import uuid

import pytest

from backend.database.connection import Database, DatabaseConfigurationError, normalize_database_url

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def test_only_postgresql_urls_accepted():
    with pytest.raises(DatabaseConfigurationError):
        normalize_database_url("sqlite:///promptshield.db")
    with pytest.raises(DatabaseConfigurationError):
        normalize_database_url("")
    assert normalize_database_url("postgres://u:p@h:5432/db").startswith("postgresql+psycopg://")
    assert normalize_database_url("postgresql://u:p@h/db").startswith("postgresql+psycopg://")


@pytest.fixture
def pg_repository():
    from sqlalchemy import text

    from backend.database.repository import ChatRepository

    schema = f"ps_test_{uuid.uuid4().hex[:10]}"
    admin = Database(TEST_DATABASE_URL)
    with admin.engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    db = Database(TEST_DATABASE_URL, schema=schema)
    db.create_schema()
    try:
        yield ChatRepository(db)
    finally:
        db.dispose()
        with admin.engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.mark.postgres
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")
def test_repository_round_trip(pg_repository):
    repo = pg_repository
    assert repo.ping()
    cid = repo.ensure_conversation(None)
    assert repo.ensure_conversation(cid) == cid
    assert repo.ensure_conversation("not-a-uuid") != cid

    msg_id = repo.add_message(cid, "user", "hello")
    event_id = repo.create_event(conversation_id=cid, source="chat", prompt_length=5, decision="ALLOW",
                                 classification="benign", confidence=0.9, p_injection=0.1,
                                 response_status="in_progress")
    repo.link_message(msg_id, event_id)
    repo.update_event(event_id, response_status="generated", llm_called=True)
    repo.add_message(cid, "assistant", "hi there", event_id)

    event = repo.get_event(event_id)
    assert event["prompt"] == "hello" and event["llm_called"] is True
    history = repo.allowed_history(cid, before_message_id=10**9, max_turns=5)
    assert [(t.role, t.text) for t in history] == [("user", "hello"), ("model", "hi there")]

    with pytest.raises(ValueError):
        repo.create_event(source="chat", prompt_length=1, response_status="x", api_key="nope")


@pytest.mark.postgres
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")
def test_review_claim_is_atomic(pg_repository):
    repo = pg_repository
    event_id = repo.create_event(source="chat", prompt_length=3, decision="FLAG", response_status="pending_review",
                                 review_status="pending")
    assert [e["id"] for e in repo.pending_reviews()] == [event_id]
    assert repo.claim_review(event_id, "approved") is True
    assert repo.claim_review(event_id, "approved") is False
    assert repo.pending_reviews() == []


@pytest.mark.postgres
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")
def test_security_gate_persists_to_postgres(pg_repository):
    from backend.services.security_gate import SecurityGate
    from conftest import (ALLOW_PROMPT, BLOCK_PROMPT, DEFAULT_SCORES, FLAG_PROMPT, FakeDetector, FakeGeminiSDK,
                          RecordingAudit, make_chat_service)

    sdk = FakeGeminiSDK()
    gate = SecurityGate(FakeDetector(DEFAULT_SCORES), make_chat_service(sdk), pg_repository, RecordingAudit())

    allowed = gate.process_chat(ALLOW_PROMPT, None)
    cid = allowed["conversation_id"]
    blocked = gate.process_chat(BLOCK_PROMPT, cid)
    flagged = gate.process_chat(FLAG_PROMPT, cid)
    gate.process_chat("follow-up", cid)

    assert pg_repository.get_event(allowed["event_id"])["response_status"] == "generated"
    assert pg_repository.get_event(blocked["event_id"])["response_status"] == "blocked"
    assert pg_repository.get_event(blocked["event_id"])["llm_called"] is False
    assert pg_repository.get_event(flagged["event_id"])["review_status"] == "pending"
    sent = [c.parts[0].text for c in sdk.calls[-1]["contents"]]
    assert BLOCK_PROMPT not in sent and FLAG_PROMPT not in sent
    assert len(sdk.calls) == 2
