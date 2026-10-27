"""
PromptShield - Persistence Repository
All database reads and writes used by the application go through this class,
so route handlers and the security gate never build SQL themselves.

Methods return plain Python values (ids, dicts, ChatTurn lists), never
ORM objects bound to a closed session.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import select, update

from backend.database.connection import Database
from backend.database.models import ChatMessage, Conversation, DetectionEvent
from llm.prompt_templates import MODEL_ROLE, USER_ROLE, ChatTurn

# response_status values
GENERATED = "generated"
LLM_ERROR = "llm_error"
PENDING_REVIEW = "pending_review"
REJECTED_BY_REVIEW = "rejected_by_review"
BLOCKED = "blocked"
DETECTOR_ERROR = "detector_error"
DETECTION_ONLY = "detection_only"
IN_PROGRESS = "in_progress"

_EVENT_FIELDS = {c.name for c in DetectionEvent.__table__.columns} - {"id", "created_at"}


def _parse_uuid(value: Optional[str]) -> Optional[uuid.UUID]:
    if not value:
        return None
    try:
        return uuid.UUID(str(value))
    except ValueError:
        return None


class ChatRepository:
    def __init__(self, database: Database):
        self.db = database

    def ping(self) -> bool:
        return self.db.ping()

    def ensure_conversation(self, conversation_id: Optional[str]) -> str:
        """Return an existing conversation id, or create a new conversation."""
        wanted = _parse_uuid(conversation_id)
        with self.db.session() as s:
            if wanted is not None and s.get(Conversation, wanted) is not None:
                return str(wanted)
            conv = Conversation()
            s.add(conv)
            s.flush()
            return str(conv.id)

    def add_message(self, conversation_id: str, role: str, content: str, event_id: Optional[int] = None) -> int:
        with self.db.session() as s:
            msg = ChatMessage(conversation_id=uuid.UUID(conversation_id), role=role, content=content,
                              detection_event_id=event_id)
            s.add(msg)
            s.flush()
            return msg.id

    def link_message(self, message_id: int, event_id: int) -> None:
        with self.db.session() as s:
            s.get(ChatMessage, message_id).detection_event_id = event_id

    def create_event(self, **fields) -> int:
        unknown = set(fields) - _EVENT_FIELDS
        if unknown:
            raise ValueError(f"Unknown detection event fields: {sorted(unknown)}")
        if fields.get("conversation_id"):
            fields["conversation_id"] = uuid.UUID(fields["conversation_id"])
        with self.db.session() as s:
            event = DetectionEvent(**fields)
            s.add(event)
            s.flush()
            return event.id

    def update_event(self, event_id: int, **fields) -> None:
        unknown = set(fields) - _EVENT_FIELDS
        if unknown:
            raise ValueError(f"Unknown detection event fields: {sorted(unknown)}")
        with self.db.session() as s:
            event = s.get(DetectionEvent, event_id)
            for key, value in fields.items():
                setattr(event, key, value)

    def allowed_history(self, conversation_id: str, before_message_id: int, max_turns: int) -> List[ChatTurn]:
        """
        Earlier turns of the conversation whose exchange was authorised and
        answered. Blocked, flagged and failed prompts are never replayed to the LLM.
        """
        with self.db.session() as s:
            rows = s.execute(
                select(ChatMessage.role, ChatMessage.content)
                .join(DetectionEvent, ChatMessage.detection_event_id == DetectionEvent.id)
                .where(
                    ChatMessage.conversation_id == uuid.UUID(conversation_id),
                    ChatMessage.id < before_message_id,
                    DetectionEvent.response_status == GENERATED,
                )
                .order_by(ChatMessage.id)
            ).all()
        turns = [ChatTurn(USER_ROLE if role == "user" else MODEL_ROLE, content) for role, content in rows]
        return turns[-2 * max_turns:] if max_turns > 0 else []

    def get_event(self, event_id: int) -> Optional[dict]:
        """Event fields plus the user prompt and its message id (for the review workflow)."""
        with self.db.session() as s:
            event = s.get(DetectionEvent, event_id)
            if event is None:
                return None
            user_msg = s.execute(
                select(ChatMessage).where(ChatMessage.detection_event_id == event_id, ChatMessage.role == "user")
            ).scalar_one_or_none()
            data = {c: getattr(event, c) for c in _EVENT_FIELDS | {"id", "created_at"}}
            data["conversation_id"] = str(event.conversation_id) if event.conversation_id else None
            data["prompt"] = user_msg.content if user_msg else None
            data["message_id"] = user_msg.id if user_msg else None
            return data

    def pending_reviews(self, limit: int = 50) -> List[dict]:
        with self.db.session() as s:
            ids = s.execute(
                select(DetectionEvent.id)
                .where(DetectionEvent.review_status == "pending")
                .order_by(DetectionEvent.id)
                .limit(limit)
            ).scalars().all()
        return [self.get_event(i) for i in ids]

    def claim_review(self, event_id: int, status: str) -> bool:
        """
        Atomically move a pending review to `status`. Returns False if the event
        is not pending (already reviewed, or never flagged), so two reviewers
        cannot both trigger generation for the same prompt.
        """
        with self.db.session() as s:
            result = s.execute(
                update(DetectionEvent)
                .where(DetectionEvent.id == event_id, DetectionEvent.review_status == "pending")
                .values(review_status=status, reviewed_at=datetime.now(timezone.utc))
            )
            return result.rowcount == 1
