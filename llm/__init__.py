"""
PromptShield LLM package: Gemini response generation.

Independent of ml_detector: nothing here imports the classifiers, and the
classifiers never import this package.
"""

from llm.chat_service import (
    ChatService,
    GenerationAuthorization,
    GenerationNotAuthorizedError,
    check_authorization,
)
from llm.gemini_client import (
    GeminiAuthenticationError,
    GeminiClient,
    GeminiConfig,
    GeminiConfigurationError,
    GeminiEmptyResponseError,
    GeminiError,
    GeminiOverloadedError,
    GeminiProviderError,
    GeminiRateLimitError,
    GeminiTimeoutError,
)
from llm.prompt_templates import ChatTurn

__all__ = [
    "ChatService", "GenerationAuthorization", "GenerationNotAuthorizedError", "check_authorization",
    "GeminiAuthenticationError", "GeminiClient", "GeminiConfig", "GeminiConfigurationError",
    "GeminiEmptyResponseError", "GeminiError", "GeminiOverloadedError", "GeminiProviderError", "GeminiRateLimitError",
    "GeminiTimeoutError", "ChatTurn",
]
