"""
PromptShield - Chat API Endpoint
Handles user messages through the ML security pipeline.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from app.ml.model_manager import model_manager
from app.ml.preprocess import validate_prompt
from app.llm.provider import get_llm_provider
from app.logging.logger import logger

router = APIRouter()

llm_provider = None


def get_llm():
    global llm_provider
    if llm_provider is None:
        llm_provider = get_llm_provider()
    return llm_provider


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4096)


class ChatResponse(BaseModel):
    classification: str
    confidence: float
    action: str
    response: str
    latency_ms: Optional[float] = None


@router.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Process a user message through the PromptShield security pipeline.
    
    Flow:
    1. Validate prompt
    2. Run ML classification (preprocess → embed → classify → decide)
    3. Based on action:
       - ALLOW → Forward to LLM, return response
       - FLAG → Return warning, do NOT forward to LLM
       - BLOCK → Return security message, do NOT forward to LLM
    """
    # Validate
    is_valid, error_msg = validate_prompt(request.message)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        # Run ML prediction
        result = model_manager.predict(request.message)
        
        # Log classification (without storing the prompt)
        logger.log_classification(
            classification=result.classification,
            confidence=result.confidence,
            action=result.action,
            model_version=model_manager.model_version,
            latency_ms=result.total_ms,
            prompt_length=len(request.message),
        )
        
        # Determine response based on action
        if result.action == "ALLOW":
            # Safe — forward to LLM
            llm = get_llm()
            if llm.is_available():
                response_text = llm.generate(request.message)
            else:
                response_text = "The AI assistant is currently unavailable. Please try again later."
        
        elif result.action == "FLAG":
            # Suspicious — do NOT forward to LLM
            response_text = (
                "⚠️ This request could not be confidently classified as safe. "
                "Please rephrase your request and try again."
            )
        
        elif result.action == "BLOCK":
            # Blocked — do NOT forward to LLM
            response_text = (
                "🛡️ This message was detected as a potential prompt injection "
                "attempt and was not forwarded to the AI model."
            )
        
        else:
            response_text = "An unexpected error occurred. Please try again."
        
        return ChatResponse(
            classification=result.classification,
            confidence=round(result.confidence, 4),
            action=result.action,
            response=response_text,
            latency_ms=round(result.total_ms, 2),
        )
    
    except Exception as e:
        logger.log_error("prediction_error", str(e))
        raise HTTPException(status_code=500, detail="Internal classification error.")
