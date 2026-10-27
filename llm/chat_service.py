"""
PromptShield - Chat Service
Generates a response for a prompt that the backend's security gate has
already authorised. It does not classify prompts and cannot change a decision.

Defence in depth: generate_reply() requires a GenerationAuthorization and
refuses anything other than
  * an ALLOW decision, or
  * a FLAG decision that a human reviewer has explicitly approved.
A BLOCK decision can never reach Gemini through this service.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

from llm.gemini_client import GeminiClient
from llm.prompt_templates import SYSTEM_INSTRUCTION, ChatTurn, build_conversation


class GenerationNotAuthorizedError(PermissionError):
    """Raised when the service is asked to generate without a valid authorisation."""


@dataclass(frozen=True)
class GenerationAuthorization:
    """Issued by the backend security gate after the decision policy has run."""
    action: str                  # decision-policy action: "ALLOW" or "FLAG"
    event_id: Optional[int] = None
    human_approved: bool = False


def check_authorization(authorization) -> None:
    if not isinstance(authorization, GenerationAuthorization):
        raise GenerationNotAuthorizedError("A GenerationAuthorization is required.")
    if authorization.action == "ALLOW":
        return
    if authorization.action == "FLAG" and authorization.human_approved:
        return
    raise GenerationNotAuthorizedError(
        f"Decision '{authorization.action}' (human_approved={authorization.human_approved}) "
        "does not permit response generation."
    )


class ChatService:
    def __init__(self, client: GeminiClient, system_instruction: str = SYSTEM_INSTRUCTION):
        self.client = client
        self.system_instruction = system_instruction
        # Outcome of the most recent real request ("ok" or an exception class name), so
        # health checks can report provider state without making a request themselves.
        self.last_outcome: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return self.client.is_configured

    @property
    def model_name(self) -> str:
        return self.client.config.model

    def generate_reply(
        self,
        authorization: GenerationAuthorization,
        prompt: str,
        history: Sequence[ChatTurn] = (),
    ) -> str:
        """
        Return Gemini's reply. Raises GenerationNotAuthorizedError if not
        authorised, or a GeminiError subclass if the provider fails - in which
        case no reply text is invented.
        """
        check_authorization(authorization)
        turns = build_conversation(history, prompt)
        try:
            reply = self.client.generate(self.system_instruction, turns)
        except Exception as e:
            self.last_outcome = type(e).__name__
            raise
        self.last_outcome = "ok"
        return reply
