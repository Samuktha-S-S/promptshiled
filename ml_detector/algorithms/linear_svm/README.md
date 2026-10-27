# Linear Support Vector Machine

Classifies a prompt as `benign` (0) or `prompt_injection` (1) from its 384-dimensional
Sentence-BERT embedding. It was the production classifier on dataset v1; on dataset v2 the validation
rule selects Logistic Regression (see `ml_detector/models/production_model.json`).

## How it works

A linear SVM looks for the hyperplane `w · x + b = 0` that separates the two classes with the
**largest margin**, penalising points on the wrong side with the hinge loss
(`LinearSVC`, `C = 1.0`, `max_iter = 2000`). Its raw output,

```
margin(x) = w · x + b
```

is a signed distance from the hyperplane: positive means "injection side", and larger
magnitude means further from the boundary. **The margin is not a probability.**

### Turning the margin into a usable score (Platt scaling)

The ALLOW / FLAG / BLOCK policy needs a score in [0, 1]. The SVM is therefore wrapped in
`CalibratedClassifierCV(method="sigmoid", cv=3)`:

1. The training split is divided into 3 folds.
2. For each fold, a `LinearSVC` is trained on the other two folds, and a sigmoid
   `P = 1 / (1 + e^(A·margin + B))` is fitted on the held-out fold's margins (Platt scaling).
3. At prediction time the three calibrated SVMs' probabilities are averaged.

Only the training split is used for this; validation and test data are never seen during fitting.

## Why PromptShield uses it

* Maximum-margin training is robust in high-dimensional spaces like 384-d embeddings.
* Very fast at inference (one dot product per fold).
* With Platt scaling it produced the best-calibrated score of the three algorithms on validation.

## Files

| File | Purpose |
|---|---|
| `train.py` | `build_model()` (LinearSVC + Platt scaling); trains, selects thresholds on validation, saves |
| `predict.py` | `load()`, `predict()` and `decision_scores()` – returns labels, the calibrated probability **and** the raw margin |
| `evaluate.py` | Evaluates the saved model on the held-out test split |

## Commands (run from the project root)

```bash
python -m ml_detector.algorithms.linear_svm.train
python -m ml_detector.algorithms.linear_svm.predict "Ignore all previous instructions" "What is Docker?"
python -m ml_detector.algorithms.linear_svm.evaluate
```

`predict` prints both `platt_calibrated_probability=…` and `svm_margin=…` so the two can be
inspected separately.

## Artifacts

* `ml_detector/models/linear_svm.joblib` – the fitted `CalibratedClassifierCV`
* `ml_detector/models/linear_svm.json` – hyperparameters, seed, embedding info, thresholds,
  validation metrics and calibration
* `ml_detector/evaluation/results/linear_svm_test_evaluation.json`,
  `ml_detector/evaluation/figures/cm_linear_svm.png`

## Calibration (measured on the 57-prompt validation set, dataset v2)

ECE 0.089, Brier 0.026. With only 57 validation prompts these estimates are noisy.

## Measured results (held-out test set)

| Accuracy | Precision | Recall | F1 | FPR | FNR |
|---|---|---|---|---|---|
| 0.9125 | 1.0000 | 0.8511 | 0.9195 | 0.0000 | 0.1489 |

Dataset v2, 80 test prompts, `class_weight="balanced"`: 7 missed attacks, no benign prompt misclassified.
(Dataset v1: accuracy 0.9697, recall 0.9535; the v1 model was production but BLOCKed greetings such as "hi".)

## Limitations

* Linear decision boundary – cannot model interactions the embedding does not already make linear.
* The reported probability depends on Platt scaling fitted on small training folds (≈87 prompts each).
* The raw margin is averaged over the 3 fold models and is not comparable across retrains.
* On dataset v2 its recall on unseen attacks is lower than Logistic Regression's (see above).
