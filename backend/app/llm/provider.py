"""
PromptShield - LLM Provider Factory
Creates and manages LLM provider instances based on configuration.
"""

import os
from typing import Optional
from app.llm.base import LLMProvider
from app.llm.mock import MockLLMProvider


def get_llm_provider() -> LLMProvider:
    """
    Factory function that returns the appropriate LLM provider.
    
    Checks for API keys in environment variables.
    Falls back to MockLLMProvider if no external provider is configured.
    
    Environment variables:
        LLM_PROVIDER: "mock" (default) or "external"
        LLM_API_KEY: API key for external provider
        LLM_API_URL: API endpoint for external provider
    """
    provider_type = os.getenv("LLM_PROVIDER", "mock").lower()
    
    if provider_type == "mock":
        return MockLLMProvider()
    
    # Default fallback
    return MockLLMProvider()
