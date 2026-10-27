"""
PromptShield - FastAPI Application
Main entry point for the backend API server.

Run from the project root:
    python -m uvicorn backend.main:app --reload --port 8000

Startup loads the configured production classifier, connects to PostgreSQL
and prepares the Gemini client. No model training happens here.
"""

from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.chat_routes import router as chat_router
from backend.api.detection_routes import router as detection_router
from backend.core.audit_log import logger
from backend.core.config import Settings, environment_warnings, get_settings
from backend.database.connection import DB_STATUS_MESSAGES
from backend.services.security_gate import GateError, SecurityGate
from llm.chat_service import ChatService
from llm.gemini_client import GeminiClient, GeminiConfig
from ml_detector.inference.detector import PromptInjectionDetector


@dataclass
class Components:
    """Everything the gate needs; tests construct this with fakes."""
    detector: object
    chat_service: ChatService
    repository: Optional[object]
    database: Optional[object] = None          # DatabaseService (diagnostics + reconnect)
    detector_error: Optional[str] = None       # safe description of a model-loading failure
    config_warnings: List[str] = field(default_factory=list)


DETECTOR_LOAD_MESSAGE = (
    "The ML detector could not be loaded; check that the trained models exist in ml_detector/models."
)


def build_components(settings: Settings) -> Components:
    from backend.database.connection import DatabaseService
    from backend.database.repository import ChatRepository

    warnings = environment_warnings()
    for warning in warnings:
        logger.log_warning("configuration_warning", warning)

    detector, detector_error = PromptInjectionDetector(), None
    try:
        detector.load()
        logger.log_startup(model_version=detector.model_version, thresholds=detector.policy.get_thresholds())
    except Exception as e:
        # The API still starts, but every chat/detect request fails closed (503).
        detector_error = DETECTOR_LOAD_MESSAGE
        logger.log_error("detector_load_failed", f"{type(e).__name__}: {e}")

    database = DatabaseService(settings.database_url, logger=logger)
    status = database.connect()
    if status.ok:
        print(f"[PROMPTSHIELD] Connected to PostgreSQL at {database.safe_url}")
    else:
        print(f"[PROMPTSHIELD] PostgreSQL unavailable ({status.category}): {status.message}")

    chat_service = ChatService(GeminiClient(GeminiConfig.from_env()))
    return Components(detector=detector, chat_service=chat_service, repository=ChatRepository(database),
                      database=database, detector_error=detector_error, config_warnings=warnings)


def _database_health(comps: Components) -> dict:
    if comps.database is not None:
        status = comps.database.check()
        return {"ok": status.ok, "status": status.category, "message": status.message}
    if comps.repository is None:
        return {"ok": False, "status": "not_configured", "message": DB_STATUS_MESSAGES["not_configured"]}
    ok = bool(comps.repository.ping())
    return {"ok": ok, "status": "ok" if ok else "unreachable",
            "message": DB_STATUS_MESSAGES["ok" if ok else "unreachable"]}


def _llm_health(chat_service: ChatService) -> dict:
    if not chat_service.is_configured:
        return {"configured": False, "status": "not_configured", "last_request": None,
                "message": "GEMINI_API_KEY is not set; allowed prompts will not receive AI responses."}
    last = chat_service.last_outcome
    return {
        "configured": True,
        "status": "configured" if last in (None, "ok") else "last_request_failed",
        "last_request": last,
        "message": ("API key configured; Gemini is only contacted when an allowed prompt is sent."
                    if last is None else "Last Gemini request succeeded." if last == "ok"
                    else "The last Gemini request failed; see the backend log."),
    }


def create_app(components: Optional[Components] = None, settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("[PROMPTSHIELD] Starting up...")
        comps = components or build_components(settings)
        app.state.components = comps
        app.state.gate = SecurityGate(
            detector=comps.detector, chat_service=comps.chat_service, repository=comps.repository,
            audit=logger, history_turns=settings.history_turns, max_prompt_length=settings.max_prompt_length,
        )
        yield
        if comps.database is not None:
            comps.database.dispose()
        print("[PROMPTSHIELD] Shutting down...")

    app = FastAPI(
        title="PromptShield API",
        description="Semantic ML-Based Prompt Injection Detection with a Gemini-backed assistant",
        version="2.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "X-Admin-Key"],
    )

    @app.exception_handler(GateError)
    async def gate_error_handler(request: Request, exc: GateError):
        return JSONResponse(status_code=exc.status_code,
                            content={"detail": exc.detail, "code": exc.code, "category": exc.category})

    @app.exception_handler(Exception)
    async def unexpected_error_handler(request: Request, exc: Exception):
        logger.log_error("unhandled_error", type(exc).__name__)
        return JSONResponse(status_code=503,
                            content={"detail": "The request could not be processed.", "code": "internal_error"})

    app.include_router(chat_router)
    app.include_router(detection_router)

    @app.get("/api/health")
    def health(request: Request):
        """
        Live state of every dependency. Database and ML detector are required:
        if either fails, status is "degraded". Gemini is optional for health and is
        never called from here; its configuration and last outcome are reported.
        """
        comps = request.app.state.components
        detector_loaded = bool(comps.detector is not None and comps.detector.is_loaded)
        database = _database_health(comps)
        return {
            "status": "healthy" if detector_loaded and database["ok"] else "degraded",
            "model_loaded": detector_loaded,
            "model_version": comps.detector.model_version if detector_loaded else None,
            "classifier": comps.detector.algorithm if detector_loaded else None,
            "database_connected": database["ok"],
            "llm_configured": comps.chat_service.is_configured,
            "llm_model": comps.chat_service.model_name,
            "components": {
                "database": {**database, "required": True},
                "detector": {
                    "ok": detector_loaded, "required": True,
                    "status": "ok" if detector_loaded else "unavailable",
                    "message": "ML detector loaded." if detector_loaded
                    else (comps.detector_error or DETECTOR_LOAD_MESSAGE),
                },
                "llm": {**_llm_health(comps.chat_service), "required": False},
            },
            "config_warnings": comps.config_warnings,
        }

    return app


app = create_app()
