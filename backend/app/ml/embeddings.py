"""
PromptShield - Semantic Embedding Module
Generates semantic embeddings using sentence-transformers/all-MiniLM-L6-v2.
"""

import numpy as np
from typing import List, Union, Optional
import time


class SemanticEmbedder:
    """
    Generates semantic embeddings using all-MiniLM-L6-v2.
    
    This model produces 384-dimensional embeddings that capture
    semantic meaning of text, enabling the classifier to detect
    attacks even when wording is changed through paraphrasing
    or obfuscation.
    """
    
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DIM = 384
    
    def __init__(self):
        self.model = None
        self._load_time_ms = 0.0
    
    def load(self):
        """Load the sentence-transformer model."""
        from sentence_transformers import SentenceTransformer
        start = time.time()
        self.model = SentenceTransformer(self.MODEL_NAME)
        self._load_time_ms = (time.time() - start) * 1000
        print(f"[SemanticEmbedder] Model loaded in {self._load_time_ms:.0f}ms")
    
    def embed(self, text: str) -> tuple[np.ndarray, float]:
        """
        Embed a single text string.
        
        Returns:
            (embedding_vector, latency_ms)
        """
        if self.model is None:
            self.load()
        
        start = time.time()
        embedding = self.model.encode(
            text, 
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True
        )
        latency_ms = (time.time() - start) * 1000
        
        return embedding.astype(np.float32), latency_ms
    
    def embed_batch(self, texts: List[str], batch_size: int = 64) -> tuple[np.ndarray, float]:
        """
        Embed a batch of texts.
        
        Returns:
            (embeddings_array [N, 384], total_latency_ms)
        """
        if self.model is None:
            self.load()
        
        start = time.time()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        latency_ms = (time.time() - start) * 1000
        
        return embeddings.astype(np.float32), latency_ms


# Singleton instance
embedder = SemanticEmbedder()
