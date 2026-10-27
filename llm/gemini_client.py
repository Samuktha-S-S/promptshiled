"""
PromptShield - Gemini Client
Thin wrapper around Google's official Gen AI SDK (`google-genai`).

Responsibilities: client initialisation, model selection, request execution,
response extraction, timeouts and mapping provider failures to controlled
exceptions. It knows nothing about prompt-injection detection; the decision
whether a prompt may be sent is made before this module is reached.

The API key is read on the backend only (GEMINI_API_KEY) and is redacted from
every error message this module raises.
"""

import os
from dataclasses import dataclass
from typing import Optional, Sequence

from llm.prompt_templates import ChatTurn

DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_TIMEOUT_SECONDS = 30.0
# Transient provider errors (rate limit, overload) are retried once by the SDK with
# backoff. 504 (deadline expired) is NOT retried: it already took the full timeout,
# and retrying would multiply the user's wait.
RETRY_ATTEMPTS = 2
RETRY_STATUS_CODES = [429, 500, 502, 503]


class GeminiError(Exception):
    """Base class for controlled Gemini failures. Messages never contain the API key."""
    user_message = "The AI assistant is currently unavailable. Please try again later."


class GeminiConfigurationError(GeminiError):
    user_message = "The AI assistant is not configured on the server."


class GeminiAuthenticationError(GeminiError):
    user_message = "The AI assistant could not authenticate with its provider."


class GeminiRateLimitError(GeminiError):
    user_message = "The AI assistant is receiving too many requests. Please wait a moment and try again."


class GeminiTimeoutError(GeminiError):
    user_message = "The AI assistant took too long to respond. Please try again."


class GeminiOverloadedError(GeminiError):
    user_message = "The AI model is temporarily overloaded. Please try again in a moment."


class GeminiProviderError(GeminiError):
    pass


class GeminiEmptyResponseError(GeminiError):
    user_message = "The AI assistant returned no answer for this request."


@dataclass(frozen=True)
class GeminiConfig:
    api_key: Optional[str]
    model: str = DEFAULT_MODEL
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    fallback_model: Optional[str] = None  # tried once if `model` is overloaded, rate-limited or times out

    @classmethod
    def from_env(cls) -> "GeminiConfig":
        return cls(
            api_key=os.getenv("GEMINI_API_KEY") or None,
            model=os.getenv("GEMINI_MODEL") or DEFAULT_MODEL,
            timeout_seconds=float(os.getenv("GEMINI_TIMEOUT_SECONDS") or DEFAULT_TIMEOUT_SECONDS),
            fallback_model=os.getenv("GEMINI_FALLBACK_MODEL") or None,
        )

    def __repr__(self) -> str:  # never print the key
        return (f"GeminiConfig(model={self.model!r}, fallback_model={self.fallback_model!r}, "
                f"timeout_seconds={self.timeout_seconds}, api_key=***)")


class GeminiClient:
    """Executes generate_content requests against the Gemini API."""

    def __init__(self, config: GeminiConfig, sdk_client=None):
        self.config = config
        self._sdk_client = sdk_client
        self.last_model_used: Optional[str] = None

    @property
    def is_configured(self) -> bool:
        return bool(self.config.api_key) or self._sdk_client is not None

    def _client(self):
        if self._sdk_client is None:
            if not self.config.api_key:
                raise GeminiConfigurationError("GEMINI_API_KEY is not set.")
            from google import genai
            from google.genai import types

            self._sdk_client = genai.Client(
                api_key=self.config.api_key,
                http_options=types.HttpOptions(
                    timeout=int(self.config.timeout_seconds * 1000),
                    retry_options=types.HttpRetryOptions(
                        attempts=RETRY_ATTEMPTS, initial_delay=1.0, max_delay=8.0,
                        http_status_codes=RETRY_STATUS_CODES,
                    ),
                ),
            )
        return self._sdk_client

    def _redact(self, message: str) -> str:
        key = self.config.api_key
        return message.replace(key, "***") if key else message

    def _request(self, model: str, contents, config):
        import httpx
        from google.genai import errors

        try:
            return self._client().models.generate_content(model=model, contents=contents, config=config)
        except GeminiError:
            raise
        except errors.APIError as e:
            code = getattr(e, "code", None)
            detail = self._redact(f"Gemini API error {code} ({model}): {getattr(e, 'message', '') or e}")
            if code == 429:
                raise GeminiRateLimitError(detail) from None
            if code in (401, 403):
                raise GeminiAuthenticationError(detail) from None
            if code == 400 and "API key" in str(e):
                raise GeminiAuthenticationError(detail) from None
            if code in (408, 504):
                raise GeminiTimeoutError(detail) from None
            if code in (500, 502, 503):
                raise GeminiOverloadedError(detail) from None
            raise GeminiProviderError(detail) from None
        except (httpx.TimeoutException, TimeoutError):
            raise GeminiTimeoutError(f"Gemini request exceeded {self.config.timeout_seconds}s.") from None
        except Exception as e:
            raise GeminiProviderError(self._redact(f"Gemini request failed: {type(e).__name__}: {e}")) from None

    def generate(self, system_instruction: str, turns: Sequence[ChatTurn]) -> str:
        """Send the conversation to Gemini and return the response text."""
        from google.genai import types

        if not turns:
            raise ValueError("At least one conversation turn is required.")

        contents = [types.Content(role=t.role, parts=[types.Part(text=t.text)]) for t in turns]
        config = types.GenerateContentConfig(system_instruction=system_instruction)

        models = [self.config.model]
        if self.config.fallback_model and self.config.fallback_model != self.config.model:
            models.append(self.config.fallback_model)

        for index, model in enumerate(models):
            try:
                response = self._request(model, contents, config)
                self.last_model_used = model
                break
            except (GeminiOverloadedError, GeminiRateLimitError, GeminiTimeoutError):
                if index == len(models) - 1:
                    raise
                # primary model busy or too slow: try the configured fallback model once

        try:
            text = response.text
        except Exception:
            text = None
        if not text or not text.strip():
            raise GeminiEmptyResponseError("Gemini returned an empty or blocked response.")
        return text.strip()
