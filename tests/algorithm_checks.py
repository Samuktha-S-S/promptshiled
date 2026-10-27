"""Checks shared by test_logistic_regression / test_linear_svm / test_random_forest."""

import json
import os

import numpy as np
import pytest

from ml_detector.algorithms.common import ArtifactError, load_artifact
from ml_detector.config import LABELS


def check_train_and_predict(train_module, predict_module, tmp_path, splits, embedder):
    from ml_detector.algorithms.common import train_and_save

    algorithm = predict_module.ALGORITHM
    train_and_save(algorithm, train_module.build_model, predict_module.predict,
                              train_module.HYPERPARAMETERS, models_dir=str(tmp_path), splits=splits,
                              embedder=embedder)

    assert os.path.exists(tmp_path / f"{algorithm}.joblib")
    saved = json.loads((tmp_path / f"{algorithm}.json").read_text())
    assert saved["algorithm"] == algorithm
    assert saved["labels"] == list(LABELS)
    assert saved["embedding"] == embedder.info()
    assert saved["random_seed"] == 42
    assert saved["thresholds"]["selected_on"] == "validation"
    assert saved["validation_metrics"]["samples"] == len(splits["val"].y)

    model, meta = predict_module.load(str(tmp_path), embedder.info())
    out = predict_module.predict(model, splits["test"].X)
    assert out.labels.shape == (len(splits["test"].y),)
    assert set(np.unique(out.labels)) <= {0, 1}
    assert np.all((out.p_injection >= 0) & (out.p_injection <= 1))
    assert np.array_equal(out.labels, (out.p_injection >= 0.5).astype(int))
    assert out.score_type == predict_module.SCORE_TYPE == meta["score_type"]
    # separable synthetic data: the classifier must learn it
    assert (out.labels == splits["test"].y).mean() >= 0.9
    return model, out


def check_reproducible(train_module, splits):
    a = train_module.build_model(42).fit(splits["train"].X, splits["train"].y)
    b = train_module.build_model(42).fit(splits["train"].X, splits["train"].y)
    assert np.allclose(a.predict_proba(splits["test"].X), b.predict_proba(splits["test"].X))


def check_missing_artifact(predict_module, tmp_path):
    with pytest.raises(ArtifactError, match="No trained"):
        predict_module.load(str(tmp_path))


def check_incompatible_embedding(predict_module, tmp_path, embedder):
    other = {**embedder.info(), "model_name": "some-other-embedding-model"}
    with pytest.raises(ArtifactError, match="embedding"):
        predict_module.load(str(tmp_path), other)


def check_corrupt_artifact(predict_module, tmp_path):
    algorithm = predict_module.ALGORITHM
    (tmp_path / f"{algorithm}.joblib").write_bytes(b"not a model")
    (tmp_path / f"{algorithm}.json").write_text("{}")
    with pytest.raises(ArtifactError):
        load_artifact(algorithm, str(tmp_path))


def check_evaluate(evaluate_module, tmp_path, splits, embedder):
    from ml_detector.algorithms.common import evaluate_saved

    results = evaluate_saved(evaluate_module.ALGORITHM, evaluate_module.predict, models_dir=str(tmp_path),
                             splits=splits, embedder=embedder, save=False)
    m = results["metrics"]
    for key in ("accuracy", "precision", "recall", "f1", "false_positive_rate", "false_negative_rate"):
        assert 0.0 <= m[key] <= 1.0
    assert m["tn"] + m["fp"] + m["fn"] + m["tp"] == len(splits["test"].y)
    decisions = results["decisions"]
    assert sum(sum(v.values()) for v in decisions.values()) == len(splits["test"].y)
    return results
