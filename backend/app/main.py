"""
PromptShield - FastAPI Application
Main entry point for the backend API server.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.api.chat import router as chat_router
from app.ml.model_manager import model_manager
from app.logging.logger import logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML model on startup."""
    print("[PROMPTSHIELD] Starting up...")
    try:
        model_manager.load()
        logger.log_startup(
            model_version=model_manager.model_version,
            thresholds=model_manager.classifier.decision_engine.get_thresholds()
            if model_manager.classifier else {}
        )
        print("[PROMPTSHIELD] Model loaded successfully!")
    except FileNotFoundError as e:
        print(f"[WARNING] {e}")
        print("   The API will start but predictions will fail.")
        print("   Run 'python -m app.ml.train' first.")
    
    yield
    
    print("[PROMPTSHIELD] Shutting down...")


app = FastAPI(
    title="PromptShield API",
    description="Semantic ML-Based Prompt Injection Detection",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(chat_router)


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded,
        "model_version": model_manager.model_version if model_manager.is_loaded else None,
    }
