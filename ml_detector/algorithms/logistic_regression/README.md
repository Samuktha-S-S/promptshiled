# Logistic Regression

Classifies a prompt as `benign` (0) or `prompt_injection` (1) from its 384-dimensional
Sentence-BERT embedding.

## How it works

Logistic regression learns one weight per embedding dimension plus a bias. For an embedding
**x** it computes

```
z = w · x + b
P(prompt_injection | x) = 1 / (1 + e^(-z))
```

Training minimises log-loss (cross-entropy) with L2 regularisation (`C = 1.0`, the inverse
regularisation strength) using the L-BFGS solver. The decision boundary is a hyperplane in
embedding space, so the model is effectively asking: *which side of a learned direction in
meaning-space does this prompt fall on?*

## Why PromptShield uses it

* Simple, fast and interpretable baseline: one linear layer on top of the embedding.
* Produces a probability-shaped score natively, which the ALLOW / FLAG / BLOCK policy can threshold.
* Works well when the embedding already separates the classes, as Sentence-BERT largely does here.

## How embeddings enter the classifier

```
prompt ──► preprocess_text() ──► all-MiniLM-L6-v2 ──► 384-d unit vector ──► LogisticRegression
           (NFKC, lowercase,      (ml_detector/embeddings/sentence_bert.py)
            whitespace)
```

The same embedder is used for training, evaluation and inference. The model's metadata
records the embedding model and preprocessing version, and loading fails if they differ.

## Files

| File | Purpose |
|---|---|
| `train.py` | `build_model()` + hyperparameters; trains on the shared train split, selects thresholds on validation, saves the artifact |
| `predict.py` | `load()` and `predict()` – returns labels and P(injection); CLI for ad-hoc prompts |
| `evaluate.py` | Evaluates the saved model on the held-out test split |

## Commands (run from the project root)

```bash
python -m ml_detector.algorithms.logistic_regression.train
python -m ml_detector.algorithms.logistic_regression.predict "Ignore all previous instructions" "What is Docker?"
python -m ml_detector.algorithms.logistic_regression.evaluate
```

## Artifacts

* `ml_detector/models/logistic_regression.joblib` – the fitted scikit-learn model
* `ml_detector/models/logistic_regression.json` – hyperparameters, seed (42), embedding info,
  validation-selected thresholds, validation metrics and calibration
* `ml_detector/evaluation/results/logistic_regression_test_evaluation.json` and
  `ml_detector/evaluation/figures/cm_logistic_regression.png` – written by `evaluate.py`

## Score

`predict()` returns `p_injection`, the model's logistic output. It is a probability *estimate*
but was **not** separately calibrated. Measured on the 57-prompt validation set (dataset v2):
ECE 0.161, Brier 0.060 – the scores are reasonably ranked but not well calibrated.

## Measured results (held-out test set)

| Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|
| 0.9750 | 1.0000 | 0.9574 | 0.9783 | 0.0000 | 0.0426 |

Dataset v2, 80 test prompts, `class_weight="balanced"`: 2 missed attacks (1 obfuscated, 1 "unseen"); no benign prompt
misclassified. **Current production model.** (Dataset v1: accuracy 0.9545, recall 0.9302.)
See `ml_detector/evaluation/results/comparison_report.md` for the full comparison.

## Limitations

* Linear in embedding space: attacks whose embeddings sit close to benign topics are missed.
* Scores are not calibrated probabilities (see above).
* Trained on 262 synthetic prompts (dataset v2); results on real traffic will differ.
