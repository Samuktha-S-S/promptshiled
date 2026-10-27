"""
PromptShield - Database Schema (SQLAlchemy ORM, PostgreSQL)

conversations      one row per chat session
chat_messages      chat history (user prompts and assistant replies)
detection_events   every classification: prediction, decision, LLM outcome, review state

No API keys or other secrets are stored.
"""

import uuid
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def _now() -> datetime:
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    pass


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    messages: Mapped[List["ChatMessage"]] = relationship(back_populates="conversation", order_by="ChatMessage.id")


class ChatMessage(Base):
    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # "user" | "assistant"
    content: Mapped[str] = mapped_column(Text)
    detection_event_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("detection_events.id", ondelete="SET NULL"), index=True, nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class DetectionEvent(Base):
    __tablename__ = "detection_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[Optional[uuid.UUID]] = mapped_column(
        ForeignKey("conversations.id", ondelete="SET NULL"), index=True, nullable=True
    )
    source: Mapped[str] = mapped_column(String(16))           # "chat" | "detect"
    prompt_length: Mapped[int] = mapped_column(Integer)

    # Prediction (null when the detector failed)
    algorithm: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    model_version: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    classification: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    p_injection: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    score_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Security decision and what happened next
    decision: Mapped[Optional[str]] = mapped_column(String(8), index=True, nullable=True)  # ALLOW|FLAG|BLOCK
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    llm_called: Mapped[bool] = mapped_column(Boolean, default=False)
    response_status: Mapped[str] = mapped_column(String(32), index=True)
    error_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # FLAG review workflow
    review_status: Mapped[Optional[str]] = mapped_column(String(16), index=True, nullable=True)  # pending|approved|rejected
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now, index=True)
