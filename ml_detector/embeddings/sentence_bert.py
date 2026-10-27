"""
PromptShield - Sentence-BERT Embedding Module
Generates semantic embeddings using sentence-transformers/all-MiniLM-L6-v2.

This is the ONLY place that turns text into feature vectors for the three
classifiers. Training, evaluation and inference all go through it, so the
classifiers always see vectors produced by the same model and preprocessing.

This module contains no classifier logic and never calls an LLM.
"""

import threading
import time
from typing import Optional, Sequence, Union

import numpy as np

from ml_detector.preprocessing import PREPROCESSING_VERSION, preprocess_text

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


class EmbeddingError(RuntimeError):
    """Raised when the embedding model cannot be loaded or fails to encode."""


class SentenceBertEmbedder:
    """
    Generates semantic embeddings using all-MiniLM-L6-v2.

    This model produces 384-dimensional, L2-normalised embeddings that capture
    the semantic meaning of text, enabling a classifier to recognise attacks
    even when their wording is changed through paraphrasing.

    Every input is passed through `preprocess_text` before encoding so that
    training and inference use identical text normalisation.
    """

    def __init__(self, model_name: str = MODEL_NAME, device: Optional[str] = None):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.load_time_ms = 0.0
        self._lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load the sentence-transformer model (only once per instance)."""
        with self._lock:
            if self.model is not None:
                return
            try:
                from sentence_transformers import SentenceTransformer

                start = time.perf_counter()
                self.model = SentenceTransformer(self.model_name, device=self.device)
                self.load_time_ms = (time.perf_counter() - start) * 1000
            except Exception as e:
                raise EmbeddingError(
                    f"Could not load embedding model '{self.model_name}': {type(e).__name__}"
                ) from e

    def encode(self, texts: Union[str, Sequence[str]], batch_size: int = 64) -> np.ndarray:
        """
        Embed one prompt or a batch of prompts.

        Returns:
            float32 array of shape [N, 384] (N = 1 for a single string)
        """
        if isinstance(texts, str):
            texts = [texts]
        if not isinstance(texts, (list, tuple)) or len(texts) == 0:
            raise ValueError("encode() expects a non-empty string or list of strings.")
        for t in texts:
            if not isinstance(t, str):
                raise TypeError(f"All inputs must be strings, got {type(t).__name__}.")

        processed = [preprocess_text(t) for t in texts]

        if self.model is None:
            self.load()

        try:
            embeddings = self.model.encode(
                processed,
                batch_size=batch_size,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
        except Exception as e:
            raise EmbeddingError(f"Embedding inference failed: {type(e).__name__}") from e

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != EMBEDDING_DIM:
            raise EmbeddingError(
                f"Unexpected embedding shape {embeddings.shape}; expected [N, {EMBEDDING_DIM}]."
            )
        return embeddings

    def embed(self, text: str) -> np.ndarray:
        """Embed a single prompt. Returns a float32 vector of shape [384]."""
        return self.encode(text)[0]

    def info(self) -> dict:
        """Describes the embedding pipeline; stored with every trained model."""
        return {
            "model_name": self.model_name,
            "embedding_dim": EMBEDDING_DIM,
            "normalized": True,
            "preprocessing": PREPROCESSING_VERSION,
        }


_shared_embedders: dict = {}
_shared_lock = threading.Lock()


def get_embedder(model_name: str = MODEL_NAME) -> SentenceBertEmbedder:
    """Return a process-wide embedder so the model is not reloaded repeatedly."""
    with _shared_lock:
        if model_name not in _shared_embedders:
            _shared_embedders[model_name] = SentenceBertEmbedder(model_name)
        return _shared_embedders[model_name]
