"""
Database configuration, diagnostics, reconnection and how DB failures surface in the API.

Unit tests need no server. Tests marked `postgres` use TEST_DATABASE_URL and only
create/drop throwaway schemas; they never modify existing tables.
"""

import os
import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError, ProgrammingError

from backend.core.config import Settings, environment_warnings
from backend.database.connection import (
    Database, DatabaseService, DatabaseUnavailableError, classify_db_error,
)
from conftest import ALLOW_PROMPT, FakeGeminiSDK, FakeRepository

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")


def op_error(message, sqlstate=None):
    class DriverError(Exception):
        pass

    orig = DriverError(message)
    orig.sqlstate = sqlstate
    return OperationalError("SELECT 1", {}, orig)


# --------------------------------------------------------------------------- configuration

def test_settings_load_database_url_from_env_file_regardless_of_cwd(tmp_path, monkeypatch):
    env_file = tmp_path / "custom.env"
    env_file.write_text("DATABASE_URL=postgresql://u:p%40ss@localhost:5432/promptshield\n")
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.chdir(os.path.expanduser("~"))
    settings = Settings.from_env(env_file=str(env_file))
    url = make_url(settings.database_url)
    assert (url.username, url.password, url.database) == ("u", "p@ss", "promptshield")


def test_process_environment_overrides_env_file(tmp_path, monkeypatch):
    env_file = tmp_path / "x.env"
    env_file.write_text("DATABASE_URL=postgresql://file/db\n")
    monkeypatch.setenv("DATABASE_URL", "postgresql://process/db")
    assert Settings.from_env(env_file=str(env_file)).database_url == "postgresql://process/db"


def test_settings_repr_hides_admin_key():
    assert "secret-admin" not in repr(Settings(database_url=None, admin_api_key="secret-admin"))


def test_environment_warnings(tmp_path):
    template = "DATABASE_URL=postgresql://promptshield:change-me@localhost/promptshield\n"
    assert any(".env not found" in w for w in environment_warnings(str(tmp_path)))

    (tmp_path / ".env.example").write_text(template)
    (tmp_path / ".env").write_text(template)
    assert any("unedited copy" in w for w in environment_warnings(str(tmp_path)))

    (tmp_path / "example.env").write_text("DATABASE_URL=postgresql://postgres:secret@localhost/x\n")
    warnings = environment_warnings(str(tmp_path))
    assert any("'example.env'" in w for w in warnings)
    assert not any("secret" in w for w in warnings)  # values never echoed

    (tmp_path / ".env").write_text("DATABASE_URL=postgresql://postgres:real@localhost/promptshield\n")
    (tmp_path / "example.env").unlink()
    assert environment_warnings(str(tmp_path)) == []


# --------------------------------------------------------------------------- error classification

@pytest.mark.parametrize("exc, url, expected", [
    (op_error('FATAL:  password authentication failed for user "promptshield"'),
     "postgresql://promptshield:change-me@localhost/promptshield", "placeholder_credentials"),
    (op_error('FATAL:  password authentication failed for user "postgres"'),
     "postgresql://postgres:wrong@localhost/promptshield", "authentication_failed"),
    (op_error("x", sqlstate="28P01"), None, "authentication_failed"),
    (op_error('FATAL:  database "nope" does not exist'), None, "database_missing"),
    (op_error("connection timeout expired"), None, "unreachable"),
    (op_error("failed to resolve host 'db.invalid'"), None, "unreachable"),
    (op_error('connection to server at "127.0.0.1", port 5499 failed: Connection refused'), None, "unreachable"),
    (ProgrammingError("SELECT", {}, type("UndefinedTable", (Exception,), {"sqlstate": "42P01"})("no table")),
     None, "schema_missing"),
    (ProgrammingError("SELECT", {}, type("E", (Exception,), {"sqlstate": "42501"})("permission denied")),
     None, "permission_denied"),
    (DatabaseUnavailableError("database_missing"), None, "database_missing"),
])
def test_classify_db_error(exc, url, expected):
    assert classify_db_error(exc, url) == expected


# --------------------------------------------------------------------------- DatabaseService (no server)

def test_service_without_url_reports_not_configured():
    service = DatabaseService(None)
    assert service.connect().category == "not_configured"
    assert service.check().ok is False
    with pytest.raises(DatabaseUnavailableError):
        with service.session():
            pass


def test_service_rejects_non_postgres_url():
    service = DatabaseService("sqlite:///promptshield.db")
    assert service.connect().category == "invalid_configuration"


def test_service_logs_without_password():
    class Log:
        def __init__(self):
            self.lines = []

        def log_error(self, kind, message):
            self.lines.append(message)

    log = Log()
    DatabaseService("postgresql://u:topsecretpw@127.0.0.1:1/x", logger=log).connect()
    assert log.lines and all("topsecretpw" not in line for line in log.lines)


# --------------------------------------------------------------------------- gate / API behaviour

class FailingRepository(FakeRepository):
    """Database dies after the conversation is created (e.g. server stopped mid-request)."""

    def create_event(self, **fields):
        raise op_error("server closed the connection unexpectedly")


def test_database_failure_mid_request_is_reported_and_never_calls_gemini(make_client):
    sdk = FakeGeminiSDK()
    r = make_client(repository=FailingRepository(), sdk=sdk).post("/api/chat", json={"message": ALLOW_PROMPT})
    assert r.status_code == 503
    body = r.json()
    assert body["code"] == "database_unavailable"
    assert body["category"] == "unreachable"
    assert "not processed" in body["detail"]
    assert sdk.calls == []


def test_missing_repository_reports_structured_error(make_client):
    body = make_client(repository=None).post("/api/chat", json={"message": ALLOW_PROMPT}).json()
    assert body["code"] == "database_unavailable" and body["category"] == "not_configured"


def test_health_reports_each_component(make_client):
    body = make_client().get("/api/health").json()
    comps = body["components"]
    assert body["status"] == "healthy"
    assert comps["database"] == {"ok": True, "status": "ok", "message": comps["database"]["message"], "required": True}
    assert comps["detector"]["ok"] is True
    assert comps["llm"]["configured"] is True and comps["llm"]["required"] is False
    assert comps["llm"]["status"] == "configured" and comps["llm"]["last_request"] is None


def test_health_identifies_database_as_failing_dependency(make_client):
    from backend.core.config import Settings
    from backend.main import Components, create_app
    from conftest import FakeDetector, make_chat_service
    from fastapi.testclient import TestClient

    comps = Components(detector=FakeDetector(), chat_service=make_chat_service(FakeGeminiSDK()),
                       repository=None, database=DatabaseService(None))
    with TestClient(create_app(comps, Settings(database_url=None))) as client:
        body = client.get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["database_connected"] is False
    assert body["components"]["database"]["status"] == "not_configured"
    assert body["components"]["detector"]["ok"] is True  # ML reported separately


def test_health_reports_detector_failure_separately(make_client):
    from conftest import FakeDetector

    body = make_client(detector=FakeDetector(loaded=False)).get("/api/health").json()
    assert body["status"] == "degraded"
    assert body["components"]["detector"]["ok"] is False
    assert body["components"]["database"]["ok"] is True


def test_health_llm_not_configured_is_not_a_failure(make_client):
    body = make_client(sdk=None, api_key=None).get("/api/health").json()
    assert body["status"] == "healthy"
    assert body["components"]["llm"]["status"] == "not_configured"


def test_llm_failure_reported_with_code_and_in_health(make_client):
    client = make_client(sdk=FakeGeminiSDK(error=TimeoutError()))
    chat = client.post("/api/chat", json={"message": ALLOW_PROMPT}).json()
    assert chat["llm_status"] == "error" and chat["llm_error_code"] == "llm_timeout"
    llm = client.get("/api/health").json()["components"]["llm"]
    assert llm["status"] == "last_request_failed" and llm["last_request"] == "GeminiTimeoutError"


# --------------------------------------------------------------------------- real PostgreSQL

def _with(url, **parts):
    return make_url(url).set(**parts).render_as_string(hide_password=False)


@pytest.mark.parametrize("parts, expected", [
    ({"password": "definitely-wrong-password"}, "authentication_failed"),
    ({"username": "promptshield", "password": "change-me"}, "placeholder_credentials"),
    ({"database": "ps_db_that_does_not_exist"}, "database_missing"),
])
@pytest.mark.postgres
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")
def test_real_connection_failures_are_categorised(parts, expected):
    assert DatabaseService(_with(TEST_DATABASE_URL, **parts)).connect().category == expected


@pytest.fixture
def empty_schema():
    schema = f"ps_test_{uuid.uuid4().hex[:10]}"
    admin = Database(TEST_DATABASE_URL)
    with admin.engine.begin() as conn:
        conn.execute(text(f'CREATE SCHEMA "{schema}"'))
    try:
        yield schema
    finally:
        with admin.engine.begin() as conn:
            conn.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.mark.postgres
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")
def test_real_connection_select_1_and_tables(empty_schema):
    service = DatabaseService(TEST_DATABASE_URL, schema=empty_schema)
    assert service.connect().ok  # creates the missing tables non-destructively
    status = service.check()
    assert status.ok and status.category == "ok"
    with service.session() as s:
        assert s.execute(text("SELECT 1")).scalar() == 1
    service.dispose()


@pytest.mark.postgres
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")
def test_missing_tables_distinguished_from_connection_failure(empty_schema, monkeypatch):
    monkeypatch.setattr(Database, "create_schema", lambda self: None)  # simulate no CREATE permission
    status = DatabaseService(TEST_DATABASE_URL, schema=empty_schema).connect()
    assert status.category == "schema_missing"


@pytest.mark.postgres
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")
def test_service_recovers_once_configuration_is_fixed(empty_schema):
    service = DatabaseService(_with(TEST_DATABASE_URL, password="wrong"), schema=empty_schema)
    assert service.connect().category == "authentication_failed"
    service.url = TEST_DATABASE_URL
    service.RETRY_SECONDS = 0
    assert service.check().ok  # reconnects on next use, no restart required
    service.dispose()


@pytest.mark.postgres
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="TEST_DATABASE_URL not set")
def test_failed_transaction_is_rolled_back(empty_schema):
    from backend.database.models import Conversation

    service = DatabaseService(TEST_DATABASE_URL, schema=empty_schema)
    service.connect()
    with pytest.raises(RuntimeError):
        with service.session() as s:
            s.add(Conversation())
            s.flush()
            raise RuntimeError("fail after write")
    with service.session() as s:
        assert s.execute(text("SELECT count(*) FROM conversations")).scalar() == 0
    service.dispose()
