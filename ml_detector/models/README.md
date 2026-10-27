# Model artifacts

| File | Written by | Read by |
|---|---|---|
| `logistic_regression.joblib` / `.json` | `python -m ml_detector.algorithms.logistic_regression.train` | its `predict.py`, `evaluate.py`, the detector |
| `linear_svm.joblib` / `.json` | `python -m ml_detector.algorithms.linear_svm.train` | 〃 |
| `random_forest.joblib` / `.json` | `python -m ml_detector.algorithms.random_forest.train` | 〃 |
| `production_model.json` | `python -m ml_detector.evaluation.compare` | `ml_detector/inference/detector.py` |
| `legacy/` | the original `backend/app/ml/train.py` | nothing (kept for reference) |
| `archive/v1_2026-10-26/` | the dataset-v1 models that blocked "hi" | nothing (kept for comparison) |
| `candidates/` | `train.py --models-dir ml_detector/models/candidates/<name>` | evaluated before promotion |

Paths are defined once in `ml_detector/config.py` (`MODELS_DIR`) and resolved by
`ml_detector/algorithms/common.py:artifact_paths()`, so training and inference always agree.

## Metadata (`<algorithm>.json`)

Each model is saved with: algorithm name, artifact format version, hyperparameters, random seed,
label order, embedding model + preprocessing version, **validation-selected** ALLOW/FLAG/BLOCK
thresholds, validation metrics and calibration, scikit-learn version, and training timestamp.

`load_artifact()` refuses to load a model if the files are missing, the label order differs, the
feature count differs, or the embedding model / preprocessing differs from the one in use.

## Choosing the production model

`production_model.json` names the algorithm the API uses. `compare.py` picks it using
**validation** results only (0.6·F1 + 0.4·(1−FPR), ties broken by lower Brier score).
Current selection: `logistic_regression` (dataset v2; v1 selected `linear_svm`). Override without retraining:

```bash
# PowerShell
$env:PROMPTSHIELD_CLASSIFIER="logistic_regression"
```

## Regenerating

```bash
python -m ml_detector.algorithms.logistic_regression.train
python -m ml_detector.algorithms.linear_svm.train
python -m ml_detector.algorithms.random_forest.train
python -m ml_detector.evaluation.compare
```

Training is deterministic (seed 42) for a given scikit-learn version. Artifacts are pickles: only
load files you produced yourself, and retrain after changing the scikit-learn version.

## `legacy/`

The original project's production model: **TF-IDF + Logistic Regression** (`best_model.joblib`,
`tfidf_vectorizer.joblib`, thresholds 0.55/0.55). It was chosen by a composite score computed on
the **test** set. It is kept unchanged for reference but is no longer loaded by the API, which now
always uses the Sentence-BERT pipeline. The TF-IDF approach is still evaluated as a baseline
in `compare.py`.
