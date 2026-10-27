"""Random Forest: training, prediction, artifacts and evaluation."""

import algorithm_checks as checks
from conftest import requires_trained_models
from ml_detector.algorithms.random_forest import evaluate, predict, train


def test_train_save_load_predict(tmp_path, synthetic_splits, fake_embedder):
    checks.check_train_and_predict(train, predict, tmp_path, synthetic_splits, fake_embedder)


def test_training_is_reproducible(synthetic_splits):
    checks.check_reproducible(train, synthetic_splits)


def test_missing_artifact_raises(tmp_path):
    checks.check_missing_artifact(predict, tmp_path)


def test_incompatible_embedding_rejected(tmp_path, synthetic_splits, fake_embedder):
    checks.check_train_and_predict(train, predict, tmp_path, synthetic_splits, fake_embedder)
    checks.check_incompatible_embedding(predict, tmp_path, fake_embedder)


def test_corrupt_artifact_rejected(tmp_path):
    checks.check_corrupt_artifact(predict, tmp_path)


def test_evaluate_reports_all_metrics(tmp_path, synthetic_splits, fake_embedder):
    checks.check_train_and_predict(train, predict, tmp_path, synthetic_splits, fake_embedder)
    checks.check_evaluate(evaluate, tmp_path, synthetic_splits, fake_embedder)


def test_hyperparameters_and_probabilities(tmp_path, synthetic_splits, fake_embedder):
    model, out = checks.check_train_and_predict(train, predict, tmp_path, synthetic_splits, fake_embedder)
    assert model.n_estimators == train.HYPERPARAMETERS["n_estimators"]
    assert model.max_depth == train.HYPERPARAMETERS["max_depth"]
    assert hasattr(model, "predict_proba")
    assert out.decision_scores is None
    assert out.score_type == "tree_vote_fraction"


@requires_trained_models
def test_trained_artifact_loads_and_predicts():
    from ml_detector.embeddings import get_embedder

    embedder = get_embedder()
    model, metadata = predict.load(embedding_info=embedder.info())
    out = predict.predict(model, embedder.encode(["What is the capital of France?"]))
    assert out.labels.shape == (1,)
    assert 0.0 <= out.p_injection[0] <= 1.0
    assert metadata["thresholds"]["selected_on"] == "validation"
