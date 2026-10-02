"""
PromptShield - Text Preprocessing Module
Cleans and normalizes user prompts before embedding.
"""

import re
import unicodedata
from typing import Optional


def preprocess_text(text: str) -> str:
    """
    Preprocess a prompt for ML classification.
    
    Steps:
    1. Unicode normalization (NFKC)
    2. Lowercase conversion
    3. Whitespace normalization
    4. Remove excessive special character repetition
    5. Strip leading/trailing whitespace
    
    Note: We intentionally preserve most punctuation and structure
    because the semantic embedding model benefits from natural text.
    Heavy stripping would destroy semantic meaning.
    """
    if not text or not isinstance(text, str):
        return ""
    
    # Unicode normalize
    text = unicodedata.normalize("NFKC", text)
    
    # Lowercase
    text = text.lower()
    
    # Normalize whitespace (collapse multiple spaces/newlines)
    text = re.sub(r'\s+', ' ', text)
    
    # Remove excessive repeated special characters (e.g., "!!!!!!" -> "!")
    text = re.sub(r'([!?.*#@\-=_])\1{3,}', r'\1\1', text)
    
    # Strip
    text = text.strip()
    
    return text


def validate_prompt(text: str, max_length: int = 4096) -> tuple[bool, Optional[str]]:
    """
    Validate a prompt before processing.
    
    Returns:
        (is_valid, error_message)
    """
    if not text or not isinstance(text, str):
        return False, "Empty or invalid prompt."
    
    text = text.strip()
    
    if len(text) == 0:
        return False, "Prompt is empty after trimming."
    
    if len(text) > max_length:
        return False, f"Prompt exceeds maximum length of {max_length} characters."
    
    return True, None
