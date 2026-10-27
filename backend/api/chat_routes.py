"""
PromptShield - Chat API Endpoints

POST /api/chat                   classify -> decide -> (ALLOW only) Gemini
GET  /api/review/pending         [admin] flagged requests awaiting review
POST /api/review/{event_id}      [admin] approve (then Gemini) or reject a flagged request

Route handlers are synchronous so FastAPI runs them in its thread pool; the
embedding model and the Gemini SDK call are both blocking.
"""

from fastapi import APIRouter, Depends, Request

from backend.core.security import require_admin
from backend.schemas.chat_schema import (
    ChatRequest, ChatResponse, ReviewDecisionRequest, ReviewDecisionResponse, ReviewItem, ReviewList,
)

router = APIRouter(prefix="/api")


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, request: Request):
    """Process a user message through the PromptShield security pipeline."""
    return request.app.state.gate.process_chat(body.message, body.conversation_id)


@router.get("/review/pending", response_model=ReviewList, dependencies=[Depends(require_admin)])
def pending_reviews(request: Request):
    items = request.app.state.gate.pending_reviews()
    return {"items": [ReviewItem(event_id=i["id"], **{k: i[k] for k in ReviewItem.model_fields if k != "event_id"})
                      for i in items]}


@router.post("/review/{event_id}", response_model=ReviewDecisionResponse, dependencies=[Depends(require_admin)])
def review(event_id: int, body: ReviewDecisionRequest, request: Request):
    return request.app.state.gate.review(event_id, body.approve)
