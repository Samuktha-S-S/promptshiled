"""
PromptShield - Model Manager
Loads and manages the production model for the FastAPI backend.
"""

import os
import json
import joblib
import numpy as np
from typing import Optional

from app.ml.embeddings import SemanticEmbedder
from app.ml.predict import PromptClassifier
from app.ml.decision_engine import DecisionEngine


class ModelManager:
    """
    Manages loading and serving of the trained ML model.
    Provides a single predict() interface for the API.
    """
    
    def __init__(self, models_dir: Optional[str] = None):
        if models_dir is None:
            base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.models_dir = os.path.join(base, "models")
        else:
            self.models_dir = models_dir
        
        self.classifier: Optional[PromptClassifier] = None
        self.model_version = "unknown"
        self._loaded = False
    
    def load(self):
        """Load the production model and all components."""
        print("[ModelManager] Loading model...")
        
        # Load config
        config_path = os.path.join(self.models_dir, "model_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"Model config not found at {config_path}. "
                "Run 'python -m app.ml.train' first."
            )
        
        with open(config_path) as f:
            config = json.load(f)
        
        self.model_version = config['best_model_key']
        representation = config['representation']
        label_map = {int(k): v for k, v in config['label_map'].items()}
        
        # Load model
        model = joblib.load(os.path.join(self.models_dir, "best_model.joblib"))
        
        # Load decision engine
        thresholds_path = os.path.join(self.models_dir, "thresholds.json")
        if os.path.exists(thresholds_path):
            decision_engine = DecisionEngine.load(thresholds_path)
        else:
            decision_engine = DecisionEngine()
        
        # Load embedder
        if representation == "minilm":
            embedder = SemanticEmbedder()
            embedder.load()
        else:
            # For TF-IDF, we need a wrapper that mimics the embedder interface
            tfidf = joblib.load(os.path.join(self.models_dir, "tfidf_vectorizer.joblib"))
            embedder = TfidfEmbedderWrapper(tfidf)
        
        self.classifier = PromptClassifier(
            model=model,
            embedder=embedder,
            decision_engine=decision_engine,
            label_map=label_map,
        )
        
        self._loaded = True
        print(f"[ModelManager] Model loaded: {self.model_version}")
        print(f"[ModelManager] Thresholds: {decision_engine.get_thresholds()}")
    
    def predict(self, text: str):
        """Run prediction on a single prompt."""
        if not self._loaded:
            self.load()
        return self.classifier.predict(text)
    
    @property
    def is_loaded(self) -> bool:
        return self._loaded


class TfidfEmbedderWrapper:
    """Wraps TF-IDF vectorizer to match SemanticEmbedder interface."""
    
    def __init__(self, vectorizer):
        self.vectorizer = vectorizer
    
    def embed(self, text: str):
        import time
        start = time.time()
        vec = self.vectorizer.transform([text]).toarray()[0]
        latency_ms = (time.time() - start) * 1000
        return vec.astype(np.float32), latency_ms
    
    def embed_batch(self, texts, batch_size=64):
        import time
        start = time.time()
        vecs = self.vectorizer.transform(texts).toarray()
        latency_ms = (time.time() - start) * 1000
        return vecs.astype(np.float32), latency_ms


# Singleton
model_manager = ModelManager()
