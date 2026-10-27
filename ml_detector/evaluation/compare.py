"""
PromptShield - Fair Comparison of the Three Algorithms

Usage:
    python -m ml_detector.evaluation.compare

Prerequisite: train each algorithm first, e.g.
    python -m ml_detector.algorithms.logistic_regression.train
    python -m ml_detector.algorithms.linear_svm.train
    python -m ml_detector.algorithms.random_forest.train

All three classifiers are compared on:
  * the same train / validation / test split (ml_detector/datasets/splits)
  * the same Sentence-BERT embeddings (all-MiniLM-L6-v2)
  * the same label definitions and metric code (evaluation/metrics.py)

The production classifier is selected using VALIDATION results only
(composite = 0.6 * F1 + 0.4 * (1 - FPR), the project's original formula;
ties broken by lower validation Brier score). The test set and the semantic
challenge set are reported but never used for any choice.

A TF-IDF + Logistic Regression baseline (the representation the project
originally compared against) is trained on the same split for reference.

Outputs:
    ml_detector/evaluation/results/comparison.json
    ml_detector/evaluation/results/comparison_report.md
    ml_detector/models/production_model.json
"""

import json
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml_detector.algorithms import common
from ml_detector.algorithms.linear_svm import predict as svm_predict
from ml_detector.algorithms.logistic_regression import predict as lr_predict
from ml_detector.algorithms.random_forest import predict as rf_predict
from ml_detector.config import (
    ALGORITHMS, CHALLENGE_DIR, LABELS, MODELS_DIR, PRODUCTION_CONFIG_FILE, RANDOM_SEED, RESULTS_DIR,
)
from ml_detector.datasets.loader import encode_labels, load_embedded_splits
from ml_detector.embeddings import get_embedder
from ml_detector.evaluation.calibration import evaluate_calibration
from ml_detector.evaluation.metrics import classification_metrics, per_category_metrics
from ml_detector.inference.decision_policy import DecisionPolicy
from ml_detector.preprocessing import preprocess_text

PREDICTORS = {
    "logistic_regression": lr_predict.predict,
    "linear_svm": svm_predict.predict,
    "random_forest": rf_predict.predict,
}
DISPLAY = {
    "logistic_regression": "Sentence-BERT + Logistic Regression",
    "linear_svm": "Sentence-BERT + Linear SVM",
    "random_forest": "Sentence-BERT + Random Forest",
    "tfidf_baseline": "TF-IDF + Logistic Regression (baseline)",
}
CHALLENGE_FILE = os.path.join(CHALLENGE_DIR, "semantic_challenge.csv")


def composite_score(metrics: dict) -> float:
    return 0.6 * metrics["f1"] + 0.4 * (1 - metrics["false_positive_rate"])


def policy_actions(policy: DecisionPolicy, out) -> list:
    return [
        policy.decide(LABELS[label], float(p if label == 1 else 1 - p)).action
        for label, p in zip(out.labels, out.p_injection)
    ]


def time_per_prompt_ms(fn, rows) -> dict:
    """Median / mean / p95 wall time of fn(row) over single rows (warm)."""
    fn(rows[0])  # warm-up
    times = []
    for row in rows:
        start = time.perf_counter()
        fn(row)
        times.append((time.perf_counter() - start) * 1000)
    return {"mean_ms": float(np.mean(times)), "median_ms": float(np.median(times)),
            "p95_ms": float(np.percentile(times, 95)), "runs": len(times)}


def leakage_report(splits) -> dict:
    """Exact duplicates and semantic near-duplicates between train and the evaluation splits."""
    norm = {name: set(preprocess_text(p) for p in s.frame["prompt"]) for name, s in splits.items()}
    report = {"exact_duplicates": {
        "train_val": len(norm["train"] & norm["val"]),
        "train_test": len(norm["train"] & norm["test"]),
        "val_test": len(norm["val"] & norm["test"]),
    }}
    for other in ("val", "test"):
        sims = splits[other].X @ splits["train"].X.T  # embeddings are L2-normalised
        nearest = sims.max(axis=1)
        report[f"max_cosine_{other}_to_train"] = float(nearest.max())
        report[f"{other}_prompts_with_train_neighbour_cos_ge_0.9"] = int((nearest >= 0.9).sum())
        report[f"{other}_prompts_with_train_neighbour_cos_ge_0.8"] = int((nearest >= 0.8).sum())
    return report


def run_tfidf_baseline(splits, challenge: pd.DataFrame) -> dict:
    texts = {n: [preprocess_text(p) for p in s.frame["prompt"]] for n, s in splits.items()}
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2, max_df=0.95, sublinear_tf=True)
    X_train = vectorizer.fit_transform(texts["train"])  # fitted on training data only
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0).fit(X_train, splits["train"].y)

    def predict_texts(t):
        return model.predict(vectorizer.transform(t))

    y_test_pred = predict_texts(texts["test"])
    y_chal_pred = predict_texts([preprocess_text(p) for p in challenge["prompt"]])
    return {
        "validation": classification_metrics(splits["val"].y, predict_texts(texts["val"])),
        "test": classification_metrics(splits["test"].y, y_test_pred),
        "test_per_category": per_category_metrics(splits["test"].frame, splits["test"].y, y_test_pred),
        "challenge": _challenge_summary(challenge, y_chal_pred, actions=None),
        "inference_time": {
            "feature_extraction_plus_classifier": time_per_prompt_ms(lambda t: predict_texts([t]), texts["test"]),
        },
    }


def _challenge_summary(challenge: pd.DataFrame, y_pred: np.ndarray, actions) -> dict:
    y_true = encode_labels(challenge["label"])
    by_category = {}
    for cat in challenge["category"].unique():
        mask = challenge["category"].values == cat
        entry = {"samples": int(mask.sum()), "correct": int((y_pred[mask] == y_true[mask]).sum())}
        if actions is not None:
            cat_actions = [a for a, m in zip(actions, mask) if m]
            entry["decisions"] = {k: cat_actions.count(k) for k in ("ALLOW", "FLAG", "BLOCK")}
        by_category[cat] = entry
    rows = [
        {"prompt": p, "label": l, "category": c, "predicted": LABELS[int(yp)],
         **({"decision": actions[i]} if actions is not None else {})}
        for i, (p, l, c, yp) in enumerate(zip(challenge["prompt"], challenge["label"], challenge["category"], y_pred))
    ]
    return {"metrics": classification_metrics(y_true, y_pred), "by_category": by_category, "predictions": rows}


def main(models_dir: str = MODELS_DIR, results_dir: str = RESULTS_DIR) -> dict:
    embedder = get_embedder()
    splits = load_embedded_splits(embedder)
    challenge = pd.read_csv(CHALLENGE_FILE)
    X_challenge = embedder.encode(challenge["prompt"].tolist())

    test_prompts = splits["test"].frame["prompt"].tolist()
    embedding_time = time_per_prompt_ms(embedder.embed, test_prompts)

    results = {}
    for algorithm in ALGORITHMS:
        predict = PREDICTORS[algorithm]
        model, metadata = common.load_artifact(algorithm, models_dir, embedding_info=embedder.info())
        policy = DecisionPolicy(metadata["thresholds"])

        val_out = predict(model, splits["val"].X)
        test_eval = common.evaluate_saved(algorithm, predict, models_dir=models_dir, splits=splits,
                                          embedder=embedder, save=False)
        chal_out = predict(model, X_challenge)

        results[algorithm] = {
            "display_name": DISPLAY[algorithm],
            "score_type": metadata["score_type"],
            "hyperparameters": metadata["hyperparameters"],
            "thresholds": metadata["thresholds"],
            "validation": classification_metrics(splits["val"].y, val_out.labels),
            "validation_calibration": evaluate_calibration(splits["val"].y, val_out.p_injection),
            "test": test_eval["metrics"],
            "test_per_category": test_eval["per_category"],
            "test_decisions": test_eval["decisions"],
            "challenge": _challenge_summary(challenge, chal_out.labels, policy_actions(policy, chal_out)),
            "inference_time": {
                "classifier_only": time_per_prompt_ms(lambda x: predict(model, x[None, :]), splits["test"].X),
                "embedding_shared": embedding_time,
            },
        }
        results[algorithm]["validation_composite"] = composite_score(results[algorithm]["validation"])

    baseline = run_tfidf_baseline(splits, challenge)

    ranked = sorted(
        ALGORITHMS,
        key=lambda a: (-results[a]["validation_composite"], results[a]["validation_calibration"]["brier_score"]),
    )
    production = {
        "algorithm": ranked[0],
        "selected_by": "validation composite 0.6*F1 + 0.4*(1-FPR); ties broken by lower validation Brier score",
        "validation_ranking": [
            {"algorithm": a, "composite": results[a]["validation_composite"],
             "brier_score": results[a]["validation_calibration"]["brier_score"]}
            for a in ranked
        ],
        "selected_at": datetime.now(timezone.utc).isoformat(),
        "override": "set PROMPTSHIELD_CLASSIFIER=<algorithm> to use a different trained model",
    }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {name: int(len(s.y)) for name, s in splits.items()},
        "challenge_set_size": int(len(challenge)),
        "embedding": embedder.info(),
        "leakage": leakage_report(splits),
        "algorithms": results,
        "tfidf_baseline": baseline,
        "production_selection": production,
    }

    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "comparison.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(models_dir, PRODUCTION_CONFIG_FILE), "w") as f:
        json.dump(production, f, indent=2)
    markdown = render_markdown(report)
    with open(os.path.join(results_dir, "comparison_report.md"), "w", encoding="utf-8") as f:
        f.write(markdown)

    print(markdown)
    return report


def _metric_row(name: str, m: dict, timing: str = "") -> str:
    return (f"| {name} | {m['accuracy']:.4f} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} "
            f"| {m['false_positive_rate']:.4f} | {m['false_negative_rate']:.4f} | {timing} |")


def render_markdown(r: dict) -> str:
    algos, base = r["algorithms"], r["tfidf_baseline"]
    header = ("| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | Inference time / prompt (median) |\n"
              "|---|---|---|---|---|---|---|---|")
    lines = [
        "# PromptShield - Algorithm Comparison Report",
        "",
        f"Generated: {r['generated_at']}  ",
        f"Embedding: `{r['embedding']['model_name']}` ({r['embedding']['embedding_dim']}-d, "
        f"preprocessing `{r['embedding']['preprocessing']}`)  ",
        f"Split sizes: train {r['dataset']['train']}, validation {r['dataset']['val']}, "
        f"test {r['dataset']['test']} (25 'unseen' attacks are test-only). "
        f"Semantic challenge set: {r['challenge_set_size']} prompts.",
        "",
        "All numbers below were measured by `python -m ml_detector.evaluation.compare`. "
        "The dataset is small and synthetic, so one misclassified prompt moves test accuracy by ~1.5 points; "
        "treat differences of one or two prompts as noise.",
        "",
        "## Held-out test set (positive class = prompt_injection)",
        "",
        header,
    ]
    for a, res in algos.items():
        t = res["inference_time"]
        timing = (f"{t['embedding_shared']['median_ms']:.2f} ms embedding + "
                  f"{t['classifier_only']['median_ms']:.3f} ms classifier")
        lines.append(_metric_row(res["display_name"], res["test"], timing))
    lines.append(_metric_row(DISPLAY["tfidf_baseline"], base["test"],
                             f"{base['inference_time']['feature_extraction_plus_classifier']['median_ms']:.3f} ms total"))

    lines += ["", "## Validation set (used for threshold and production-model selection)", "", header.replace(
        " Inference time / prompt (median) |", " Composite |")]
    for a, res in algos.items():
        lines.append(_metric_row(res["display_name"], res["validation"], f"{res['validation_composite']:.4f}"))
    lines.append(_metric_row(DISPLAY["tfidf_baseline"], base["validation"], "-"))

    lines += ["", "## Decision policy on the test set (ALLOW / FLAG / BLOCK counts)", "",
              "| Model | Thresholds (allow, block) | Benign prompts | Injection prompts |", "|---|---|---|---|"]
    for a, res in algos.items():
        th, d = res["thresholds"], res["test_decisions"]
        lines.append(f"| {res['display_name']} | {th['benign_allow_threshold']}, {th['injection_block_threshold']} "
                     f"| {d['benign']} | {d['prompt_injection']} |")

    lines += ["", "## Score calibration on the validation set", "",
              "| Model | Score type | ECE | Brier |", "|---|---|---|---|"]
    for a, res in algos.items():
        c = res["validation_calibration"]
        lines.append(f"| {res['display_name']} | {res['score_type']} | {c['ece']:.4f} | {c['brier_score']:.4f} |")

    lines += ["", "## Per-category test results (correct / samples)", ""]
    cats = sorted(next(iter(algos.values()))["test_per_category"])
    lines.append("| Category | " + " | ".join(DISPLAY[a] for a in algos) + " | TF-IDF baseline |")
    lines.append("|---|" + "---|" * (len(algos) + 1))
    for cat in cats:
        cells = [f"{algos[a]['test_per_category'][cat]['correct']}/{algos[a]['test_per_category'][cat]['samples']}"
                 for a in algos]
        b = base["test_per_category"][cat]
        lines.append(f"| {cat} | " + " | ".join(cells) + f" | {b['correct']}/{b['samples']} |")

    lines += ["", "## Semantic challenge set (hand-written, never used for training or selection)", ""]
    chal_cats = list(next(iter(algos.values()))["challenge"]["by_category"])
    lines.append("| Category | " + " | ".join(DISPLAY[a] for a in algos) + " | TF-IDF baseline |")
    lines.append("|---|" + "---|" * (len(algos) + 1))
    for cat in chal_cats:
        cells = []
        for a in algos:
            e = algos[a]["challenge"]["by_category"][cat]
            d = e["decisions"]
            cells.append(f"{e['correct']}/{e['samples']} (A{d['ALLOW']} F{d['FLAG']} B{d['BLOCK']})")
        b = base["challenge"]["by_category"][cat]
        lines.append(f"| {cat} | " + " | ".join(cells) + f" | {b['correct']}/{b['samples']} |")
    lines.append("")
    lines.append("Cells show correct classifications / prompts, and the policy decisions (A=ALLOW, F=FLAG, B=BLOCK).")
    for a, res in algos.items():
        m = res["challenge"]["metrics"]
        lines.append(f"- {res['display_name']}: accuracy {m['accuracy']:.4f}, recall {m['recall']:.4f}, "
                     f"FPR {m['false_positive_rate']:.4f}")
    m = base["challenge"]["metrics"]
    lines.append(f"- {DISPLAY['tfidf_baseline']}: accuracy {m['accuracy']:.4f}, recall {m['recall']:.4f}, "
                 f"FPR {m['false_positive_rate']:.4f}")

    prod = r["production_selection"]
    lines += ["", "### Challenge-set misclassifications (production model)", ""]
    misses = [p for p in algos[prod["algorithm"]]["challenge"]["predictions"] if p["predicted"] != p["label"]]
    lines += [f"- [{p['category']}] expected {p['label']}, got {p['predicted']} ({p['decision']}): \"{p['prompt']}\""
              for p in misses] or ["- none"]

    leak = r["leakage"]
    lines += ["", "## Train/test leakage check", "",
              f"- Exact duplicates after normalisation: {leak['exact_duplicates']}",
              f"- Highest cosine similarity of any test prompt to a training prompt: {leak['max_cosine_test_to_train']:.3f} "
              f"({leak['test_prompts_with_train_neighbour_cos_ge_0.9']} test prompts >= 0.9, "
              f"{leak['test_prompts_with_train_neighbour_cos_ge_0.8']} >= 0.8)",
              f"- Highest cosine similarity of any validation prompt to a training prompt: "
              f"{leak['max_cosine_val_to_train']:.3f}"]

    lines += ["", "## Production model selection", "",
              f"Selected: **{DISPLAY[prod['algorithm']]}** - {prod['selected_by']}.", ""]
    for entry in prod["validation_ranking"]:
        lines.append(f"- {DISPLAY[entry['algorithm']]}: composite {entry['composite']:.4f}, "
                     f"Brier {entry['brier_score']:.4f}")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare the trained algorithms on the shared splits.")
    parser.add_argument("--models-dir", default=MODELS_DIR, help="directory holding the trained artifacts")
    parser.add_argument("--results-dir", default=RESULTS_DIR, help="where to write the comparison report")
    args = parser.parse_args()
    main(args.models_dir, args.results_dir)
