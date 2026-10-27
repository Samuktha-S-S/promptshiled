"""
PromptShield - PostgreSQL Connection Management
The only module that creates database engines and sessions. Uses SQLAlchemy 2.x
with the psycopg (v3) driver. Only PostgreSQL URLs are accepted.
"""

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, Optional

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.exc import ArgumentError, InterfaceError, OperationalError
from sqlalchemy.orm import Session, sessionmaker


class DatabaseConfigurationError(RuntimeError):
    pass


def normalize_database_url(url: str) -> str:
    """Accept postgres:// / postgresql:// URLs and select the psycopg v3 driver."""
    if not url:
        raise DatabaseConfigurationError("DATABASE_URL is not set.")
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    parsed = make_url(url)
    if parsed.get_backend_name() != "postgresql":
        raise DatabaseConfigurationError("DATABASE_URL must point to a PostgreSQL database.")
    if parsed.drivername == "postgresql":
        parsed = parsed.set(drivername="postgresql+psycopg")
    return parsed.render_as_string(hide_password=False)


def create_db_engine(url: str, schema: Optional[str] = None) -> Engine:
    connect_args = {"connect_timeout": 5}
    if schema:
        # Used by the integration tests to work inside a throwaway schema.
        if not schema.replace("_", "").isalnum():
            raise DatabaseConfigurationError("Schema name must be alphanumeric/underscore.")
        connect_args["options"] = f"-csearch_path={schema}"
    return create_engine(
        normalize_database_url(url),
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        connect_args=connect_args,
    )


class Database:
    """Owns the engine and session factory for the application's lifetime."""

    def __init__(self, url: str, schema: Optional[str] = None):
        self.engine = create_db_engine(url, schema)
        self.session_factory = sessionmaker(bind=self.engine, expire_on_commit=False)

    def create_schema(self) -> None:
        """Create tables that do not exist yet (no destructive changes)."""
        from backend.database.models import Base
        Base.metadata.create_all(self.engine)

    def ping(self) -> bool:
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    @contextmanager
    def session(self) -> Iterator[Session]:
        """Transactional scope: commit on success, roll back on error."""
        session = self.session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def dispose(self) -> None:
        self.engine.dispose()

    @property
    def safe_url(self) -> str:
        return self.engine.url.render_as_string(hide_password=True)


# --------------------------------------------------------------------------- diagnostics

REQUIRED_TABLES = ("conversations", "chat_messages", "detection_events")
PLACEHOLDER_PASSWORDS = {"change-me"}  # the value shipped in .env.example

DB_STATUS_MESSAGES = {
    "ok": "Connected to PostgreSQL.",
    "not_checked": "The database connection has not been checked yet.",
    "not_configured": "DATABASE_URL is not set in the backend environment (.env in the project root).",
    "invalid_configuration": "DATABASE_URL is not a valid PostgreSQL connection URL.",
    "placeholder_credentials": (
        "DATABASE_URL still contains the example credentials from .env.example. "
        "Put your real PostgreSQL user and password in .env."
    ),
    "authentication_failed": "PostgreSQL rejected the username or password in DATABASE_URL.",
    "database_missing": "The database named in DATABASE_URL does not exist on the PostgreSQL server.",
    "unreachable": "The PostgreSQL server could not be reached at the configured host and port.",
    "permission_denied": "The PostgreSQL user lacks permission for the PromptShield tables.",
    "schema_missing": "Connected, but the PromptShield tables are missing and could not be created.",
    "query_failed": "Connected, but a database query failed.",
}


class DatabaseUnavailableError(RuntimeError):
    """Raised when the application needs the database and it is not usable."""

    def __init__(self, category: str):
        super().__init__(DB_STATUS_MESSAGES.get(category, DB_STATUS_MESSAGES["query_failed"]))
        self.category = category


@dataclass(frozen=True)
class DatabaseStatus:
    ok: bool
    category: str

    @property
    def message(self) -> str:
        return DB_STATUS_MESSAGES.get(self.category, DB_STATUS_MESSAGES["query_failed"])


def classify_db_error(exc: BaseException, url: Optional[str] = None) -> str:
    """Map an exception to a safe category name. Never returns exception text."""
    if isinstance(exc, DatabaseUnavailableError):
        return exc.category
    if isinstance(exc, DatabaseConfigurationError):
        return "not_configured" if "not set" in str(exc) else "invalid_configuration"
    if isinstance(exc, ArgumentError):
        return "invalid_configuration"

    orig = getattr(exc, "orig", None) or exc
    sqlstate = getattr(orig, "sqlstate", None) or ""
    text_ = str(orig).lower()

    if sqlstate in ("28P01", "28000") or "password authentication failed" in text_ \
            or "no password supplied" in text_ or "role" in text_ and "does not exist" in text_:
        return "placeholder_credentials" if _uses_placeholder_password(url) else "authentication_failed"
    if sqlstate == "3D000" or ("database" in text_ and "does not exist" in text_):
        return "database_missing"
    if sqlstate == "42P01" or sqlstate == "42703" or "undefinedtable" in type(orig).__name__.lower():
        return "schema_missing"
    if sqlstate == "42501" or "permission denied" in text_:
        return "permission_denied"
    if isinstance(exc, (OperationalError, InterfaceError)) or "timeout" in text_ or "connection" in text_ \
            or "resolve host" in text_:
        return "unreachable"
    return "query_failed"


def _uses_placeholder_password(url: Optional[str]) -> bool:
    try:
        return bool(url) and make_url(url).password in PLACEHOLDER_PASSWORDS
    except Exception:
        return False


def _safe_error_summary(exc: BaseException) -> str:
    """First line of the driver message for server logs. PostgreSQL error messages
    contain user, host and database names but never the password."""
    orig = getattr(exc, "orig", None) or exc
    return f"{type(orig).__name__}: {str(orig).splitlines()[0][:200] if str(orig) else ''}"


class DatabaseService:
    """
    Application-level database access with diagnostics and automatic reconnection.

    Exposes the same `session()` / `ping()` interface as Database, so
    ChatRepository can use it directly. If PostgreSQL is unavailable at startup
    (wrong credentials, server down), it retries at most every RETRY_SECONDS when
    the database is next needed, so the API recovers without a restart once the
    problem is fixed. Every failure is reported as a DatabaseStatus category.
    """

    RETRY_SECONDS = 5.0

    def __init__(self, url: Optional[str], logger=None, schema: Optional[str] = None):
        self.url = url
        self.logger = logger
        self.schema = schema  # only used by tests (throwaway schema)
        self.database: Optional[Database] = None
        self.status = DatabaseStatus(False, "not_checked")
        self._last_attempt = 0.0
        self._lock = threading.Lock()

    def _log_failure(self, category: str, exc: BaseException) -> None:
        if self.logger is not None:
            self.logger.log_error("database_unavailable", f"{category}: {_safe_error_summary(exc)}")

    def connect(self) -> DatabaseStatus:
        """(Re)connect, create any missing tables, and verify them."""
        with self._lock:
            self._last_attempt = time.monotonic()
            if self.database is not None:
                return self.status
            if not self.url:
                self.status = DatabaseStatus(False, "not_configured")
                return self.status
            database = None
            try:
                database = Database(self.url, self.schema)
                database.create_schema()
                missing = self._missing_tables(database)
                if missing:
                    raise DatabaseUnavailableError("schema_missing")
                self.database = database
                self.status = DatabaseStatus(True, "ok")
            except Exception as e:
                if database is not None:
                    database.dispose()
                category = classify_db_error(e, self.url)
                self.status = DatabaseStatus(False, category)
                self._log_failure(category, e)
            return self.status

    @staticmethod
    def _missing_tables(database: Database) -> list:
        existing = set(inspect(database.engine).get_table_names())
        return [t for t in REQUIRED_TABLES if t not in existing]

    def _ensure(self) -> Database:
        if self.database is None and time.monotonic() - self._last_attempt >= self.RETRY_SECONDS:
            self.connect()
        if self.database is None:
            raise DatabaseUnavailableError(self.status.category)
        return self.database

    def check(self) -> DatabaseStatus:
        """Live check used by /api/health: SELECT 1 plus the required tables."""
        try:
            database = self._ensure()
            with database.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            if self._missing_tables(database):
                raise DatabaseUnavailableError("schema_missing")
            self.status = DatabaseStatus(True, "ok")
        except Exception as e:
            category = classify_db_error(e, self.url)
            if category != self.status.category:
                self._log_failure(category, e)
            self.status = DatabaseStatus(False, category)
        return self.status

    def ping(self) -> bool:
        return self.check().ok

    @contextmanager
    def session(self) -> Iterator[Session]:
        database = self._ensure()
        with database.session() as s:
            yield s

    def dispose(self) -> None:
        if self.database is not None:
            self.database.dispose()
            self.database = None

    @property
    def safe_url(self) -> Optional[str]:
        try:
            return make_url(normalize_database_url(self.url)).render_as_string(hide_password=True)
        except Exception:
            return None
