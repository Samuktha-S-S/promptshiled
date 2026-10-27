# Random Forest

Classifies a prompt as `benign` (0) or `prompt_injection` (1) from its 384-dimensional
Sentence-BERT embedding.

## How it works

A random forest is an ensemble of decision trees. Each tree is trained on a bootstrap sample of
the training prompts and, at every split, only considers a random subset of the 384 embedding
dimensions (`sqrt(384) ≈ 19`). Each tree learns axis-aligned rules such as
"dimension 117 > 0.04 and dimension 52 ≤ −0.01 → injection". The forest's prediction averages
the trees.

## Hyperparameters

| Parameter | Value | Meaning |
|---|---|---|
| `n_estimators` | 200 | number of trees |
| `max_depth` | 20 | maximum depth of each tree |
| `max_features` | `"sqrt"` (scikit-learn default) | features considered per split |
| `random_state` | 42 | makes bootstrap sampling and feature choice reproducible |
| `n_jobs` | -1 | use all CPU cores |

These are the values used by the original project; they were **not** tuned on the test set.

## Why PromptShield uses it

* Non-linear: can capture patterns a single hyperplane cannot.
* Robust to irrelevant dimensions and needs no feature scaling.
* Provides a contrast to the two linear models in the comparison.

## Files

| File | Purpose |
|---|---|
| `train.py` | `build_model()` + hyperparameters; trains, selects thresholds on validation, saves |
| `predict.py` | `load()` and `predict()` – returns labels and the tree-vote score |
| `evaluate.py` | Evaluates the saved model on the held-out test split |

## Commands (run from the project root)

```bash
python -m ml_detector.algorithms.random_forest.train
python -m ml_detector.algorithms.random_forest.predict "Ignore all previous instructions" "What is Docker?"
python -m ml_detector.algorithms.random_forest.evaluate
```

## Artifacts

* `ml_detector/models/random_forest.joblib` (≈ 0.5 MB – 200 trees)
* `ml_detector/models/random_forest.json` – hyperparameters, seed, embedding info, thresholds, validation metrics
* `ml_detector/evaluation/results/random_forest_test_evaluation.json`,
  `ml_detector/evaluation/figures/cm_random_forest.png`

## Score

`predict_proba` returns the average over the trees of each leaf's injection fraction (close to
the share of trees voting "injection"). It is an **ensemble vote, not a calibrated probability**.
Measured on validation (dataset v2): ECE 0.231, Brier 0.094 – the least calibrated of the three. Because
votes cluster around the middle, the policy FLAGs more prompts with this model (9 of 47 test
attacks were FLAGged rather than BLOCKed).

## Measured results (held-out test set)

| Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|
| 0.9250 | 1.0000 | 0.8723 | 0.9318 | 0.0000 | 0.1277 |

Dataset v2, 80 test prompts, `class_weight="balanced"`. (Dataset v1: accuracy 0.9545, recall 0.9302.)

## Limitations

* Axis-aligned splits are a poor fit for dense embeddings, where meaning is spread across many dimensions.
* With 262 training prompts and 384 dimensions, each tree sees little data; results vary with the seed.
* Slowest classifier at inference in this project (single-prompt latency is dominated by
  dispatching work across 200 trees; see the comparison report).
