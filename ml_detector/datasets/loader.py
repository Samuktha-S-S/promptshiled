"""
PromptShield - Dataset Loading
Loads the fixed train / validation / test splits and their Sentence-BERT
embeddings. Every algorithm uses these functions, so all three classifiers are
trained and evaluated on exactly the same examples.

Embeddings are cached in datasets/processed/ together with a metadata file that
records the embedding model, preprocessing version and a hash of each split CSV.
The cache is regenerated automatically if any of those change.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from ml_detector.config import LABELS, PROCESSED_DIR, SPLITS_DIR

SPLIT_NAMES = ("train", "val", "test")
REQUIRED_COLUMNS = ("prompt", "label", "attack_type")
CACHE_META_FILE = "sbert_cache_meta.json"


class DatasetError(ValueError):
    """Raised when a split is missing or contains invalid data."""


@dataclass
class EmbeddedSplit:
    X: np.ndarray        # [N, 384] Sentence-BERT embeddings
    y: np.ndarray        # [N] int labels (0 = benign, 1 = prompt_injection)
    frame: pd.DataFrame  # original rows (prompt, label, attack_type, ...)


def encode_labels(labels) -> np.ndarray:
    """Map label strings to ints using the fixed LABELS order; reject unknown labels."""
    index = {name: i for i, name in enumerate(LABELS)}
    unknown = sorted(set(labels) - set(index))
    if unknown:
        raise DatasetError(f"Unknown labels {unknown}; expected one of {list(LABELS)}.")
    return np.array([index[l] for l in labels], dtype=np.int64)


def load_split(name: str, splits_dir: str = SPLITS_DIR) -> pd.DataFrame:
    """Load and validate one split CSV."""
    path = os.path.join(splits_dir, f"{name}.csv")
    if not os.path.exists(path):
        raise DatasetError(
            f"Split file not found: {path}. "
            "Run 'python -m ml_detector.datasets.generate' to recreate the dataset."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DatasetError(f"{path} is missing columns {missing}.")
    if df["prompt"].isna().any() or (df["prompt"].astype(str).str.strip() == "").any():
        raise DatasetError(f"{path} contains empty prompts.")
    encode_labels(df["label"])  # validates labels
    return df.reset_index(drop=True)


def load_splits(splits_dir: str = SPLITS_DIR) -> Dict[str, pd.DataFrame]:
    return {name: load_split(name, splits_dir) for name in SPLIT_NAMES}


def _file_sha256(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def load_embedded_splits(
    embedder=None,
    splits_dir: str = SPLITS_DIR,
    cache_dir: Optional[str] = PROCESSED_DIR,
) -> Dict[str, EmbeddedSplit]:
    """
    Return {split_name: EmbeddedSplit} for train, val and test.

    Args:
        embedder: a SentenceBertEmbedder (defaults to the shared instance)
        cache_dir: where to cache embeddings; None disables caching
    """
    if embedder is None:
        from ml_detector.embeddings import get_embedder
        embedder = get_embedder()

    frames = load_splits(splits_dir)
    expected_meta = {
        "embedding": embedder.info(),
        "split_hashes": {
            name: _file_sha256(os.path.join(splits_dir, f"{name}.csv")) for name in SPLIT_NAMES
        },
    }

    cached = _read_cache(cache_dir, expected_meta) if cache_dir else None
    if cached is None:
        cached = {name: embedder.encode(frames[name]["prompt"].tolist()) for name in SPLIT_NAMES}
        if cache_dir:
            _write_cache(cache_dir, expected_meta, cached)

    return {
        name: EmbeddedSplit(X=cached[name], y=encode_labels(frames[name]["label"]), frame=frames[name])
        for name in SPLIT_NAMES
    }


def _read_cache(cache_dir: str, expected_meta: dict) -> Optional[Dict[str, np.ndarray]]:
    meta_path = os.path.join(cache_dir, CACHE_META_FILE)
    if not os.path.exists(meta_path):
        return None
    with open(meta_path) as f:
        if json.load(f) != expected_meta:
            return None
    try:
        return {name: np.load(os.path.join(cache_dir, f"sbert_{name}.npy")) for name in SPLIT_NAMES}
    except OSError:
        return None


def _write_cache(cache_dir: str, meta: dict, arrays: Dict[str, np.ndarray]) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    for name, X in arrays.items():
        np.save(os.path.join(cache_dir, f"sbert_{name}.npy"), X)
    with open(os.path.join(cache_dir, CACHE_META_FILE), "w") as f:
        json.dump(meta, f, indent=2)
