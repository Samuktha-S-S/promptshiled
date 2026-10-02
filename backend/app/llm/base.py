"""
PromptShield - LLM Provider Base Interface
Abstract interface for LLM backends.
"""

from abc import ABC, abstractmethod
from typing import Optional


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Generate a response for the given prompt.
        
        Args:
            prompt: The user's message
            system_prompt: Optional system prompt for the LLM
        
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the LLM provider is available."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the name of this LLM provider."""
        pass
