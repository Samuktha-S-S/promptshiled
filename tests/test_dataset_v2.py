"""Integrity of dataset v2 (conversational extension) and its splits."""

import os

import pandas as pd

from ml_detector.config import SPLITS_DIR
from ml_detector.datasets.conversational import (
    BENIGN_CONVERSATIONAL, CONVERSATIONAL_INJECTIONS, EXTENSION_FILE, SPLITS_V1_DIR, create_extension, group_split,
    semantic_groups,
)
from ml_detector.datasets.loader import load_splits
from ml_detector.preprocessing import preprocess_text

SPLITS = ("train", "val", "test")


def test_extension_has_both_classes_and_no_duplicates():
    ext = create_extension()
    assert set(ext.label) == {"benign", "prompt_injection"}
    assert len(ext) == len(BENIGN_CONVERSATIONAL) + len(CONVERSATIONAL_INJECTIONS)
    assert not ext.prompt.str.lower().str.strip().duplicated().any()


def test_v1_rows_keep_their_original_split():
    for name in SPLITS:
        v1 = pd.read_csv(os.path.join(SPLITS_V1_DIR, f"{name}.csv"))
        v2 = pd.read_csv(os.path.join(SPLITS_DIR, f"{name}.csv"))
        assert v2.head(len(v1)).equals(v1)


def test_no_prompt_in_more_than_one_split():
    splits = load_splits()
    sets = {n: {preprocess_text(p) for p in s.prompt} for n, s in splits.items()}
    assert not sets["train"] & sets["val"]
    assert not sets["train"] & sets["test"]
    assert not sets["val"] & sets["test"]


def test_no_contradictory_labels():
    frames = pd.concat(load_splits().values())
    per_text = frames.assign(t=frames.prompt.map(preprocess_text)).groupby("t").label.nunique()
    assert (per_text == 1).all()


def test_extension_rows_assigned_by_group():
    ext = pd.read_csv(EXTENSION_FILE)
    assert ext.groupby("group").split.nunique().max() == 1  # a group never straddles splits


def test_unseen_attacks_remain_test_only():
    splits = load_splits()
    assert "unseen" not in set(splits["train"].attack_type) | set(splits["val"].attack_type)


def test_group_split_keeps_groups_together():
    import numpy as np

    frame = pd.DataFrame({"label": ["benign"] * 6 + ["prompt_injection"] * 4})
    groups = np.array([0, 0, 1, 2, 3, 4, 5, 5, 6, 7])
    assignment = group_split(frame, groups)
    for g in np.unique(groups):
        assert assignment[groups == g].nunique() == 1


def test_semantic_groups_connect_near_duplicates():
    import numpy as np

    a = np.array([1.0, 0.0])
    b = np.array([0.99, 0.141])
    b /= np.linalg.norm(b)
    c = np.array([0.0, 1.0])
    groups = semantic_groups(np.stack([a, b, c]), threshold=0.85)
    assert groups[0] == groups[1] != groups[2]
