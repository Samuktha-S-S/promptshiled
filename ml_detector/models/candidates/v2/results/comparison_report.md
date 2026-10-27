# PromptShield - Algorithm Comparison Report

Generated: 2026-10-26T18:17:39.052950+00:00  
Embedding: `sentence-transformers/all-MiniLM-L6-v2` (384-d, preprocessing `nfkc-lower-ws-v1`)  
Split sizes: train 262, validation 57, test 80 (25 'unseen' attacks are test-only). Semantic challenge set: 34 prompts.

All numbers below were measured by `python -m ml_detector.evaluation.compare`. The dataset is small and synthetic, so one misclassified prompt moves test accuracy by ~1.5 points; treat differences of one or two prompts as noise.

## Held-out test set (positive class = prompt_injection)

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | Inference time / prompt (median) |
|---|---|---|---|---|---|---|---|
| Sentence-BERT + Logistic Regression | 0.9750 | 1.0000 | 0.9574 | 0.9783 | 0.0000 | 0.0426 | 34.85 ms embedding + 0.300 ms classifier |
| Sentence-BERT + Linear SVM | 0.9125 | 1.0000 | 0.8511 | 0.9195 | 0.0000 | 0.1489 | 34.85 ms embedding + 7.850 ms classifier |
| Sentence-BERT + Random Forest | 0.9250 | 1.0000 | 0.8723 | 0.9318 | 0.0000 | 0.1277 | 34.85 ms embedding + 131.489 ms classifier |
| TF-IDF + Logistic Regression (baseline) | 0.9750 | 1.0000 | 0.9574 | 0.9783 | 0.0000 | 0.0426 | 2.178 ms total |

## Validation set (used for threshold and production-model selection)

| Model | Accuracy | Precision | Recall | F1 | FPR | FNR | Composite |
|---|---|---|---|---|---|---|---|
| Sentence-BERT + Logistic Regression | 0.9649 | 0.9565 | 0.9565 | 0.9565 | 0.0294 | 0.0435 | 0.9621 |
| Sentence-BERT + Linear SVM | 0.9474 | 0.9545 | 0.9130 | 0.9333 | 0.0294 | 0.0870 | 0.9482 |
| Sentence-BERT + Random Forest | 0.9649 | 0.9565 | 0.9565 | 0.9565 | 0.0294 | 0.0435 | 0.9621 |
| TF-IDF + Logistic Regression (baseline) | 0.9825 | 0.9583 | 1.0000 | 0.9787 | 0.0294 | 0.0000 | - |

## Decision policy on the test set (ALLOW / FLAG / BLOCK counts)

| Model | Thresholds (allow, block) | Benign prompts | Injection prompts |
|---|---|---|---|
| Sentence-BERT + Logistic Regression | 0.55, 0.55 | {'ALLOW': 33, 'FLAG': 0, 'BLOCK': 0} | {'ALLOW': 2, 'FLAG': 1, 'BLOCK': 44} |
| Sentence-BERT + Linear SVM | 0.55, 0.65 | {'ALLOW': 33, 'FLAG': 0, 'BLOCK': 0} | {'ALLOW': 5, 'FLAG': 8, 'BLOCK': 34} |
| Sentence-BERT + Random Forest | 0.55, 0.55 | {'ALLOW': 33, 'FLAG': 0, 'BLOCK': 0} | {'ALLOW': 3, 'FLAG': 9, 'BLOCK': 35} |

## Score calibration on the validation set

| Model | Score type | ECE | Brier |
|---|---|---|---|
| Sentence-BERT + Logistic Regression | logistic_probability | 0.1614 | 0.0603 |
| Sentence-BERT + Linear SVM | platt_calibrated_probability | 0.0887 | 0.0256 |
| Sentence-BERT + Random Forest | tree_vote_fraction | 0.2310 | 0.0943 |

## Per-category test results (correct / samples)

| Category | Sentence-BERT + Logistic Regression | Sentence-BERT + Linear SVM | Sentence-BERT + Random Forest | TF-IDF baseline |
|---|---|---|---|---|
| conversational | 10/10 | 10/10 | 10/10 | 10/10 |
| conversational_injection | 4/4 | 4/4 | 4/4 | 4/4 |
| direct | 2/2 | 2/2 | 2/2 | 2/2 |
| instruction_override | 3/3 | 3/3 | 3/3 | 3/3 |
| normal | 23/23 | 23/23 | 23/23 | 23/23 |
| obfuscated | 1/2 | 1/2 | 1/2 | 2/2 |
| paraphrased | 5/5 | 5/5 | 5/5 | 5/5 |
| role_based | 1/1 | 1/1 | 1/1 | 1/1 |
| system_prompt_extraction | 5/5 | 5/5 | 4/5 | 4/5 |
| unseen | 24/25 | 19/25 | 21/25 | 24/25 |

## Semantic challenge set (hand-written, never used for training or selection)

| Category | Sentence-BERT + Logistic Regression | Sentence-BERT + Linear SVM | Sentence-BERT + Random Forest | TF-IDF baseline |
|---|---|---|---|---|
| direct | 5/5 (A0 F0 B5) | 5/5 (A0 F0 B5) | 5/5 (A0 F0 B5) | 5/5 |
| paraphrased | 5/5 (A0 F0 B5) | 5/5 (A0 F0 B5) | 5/5 (A0 F0 B5) | 5/5 |
| unseen_phrasing | 6/6 (A0 F0 B6) | 6/6 (A0 F0 B6) | 6/6 (A0 F0 B6) | 6/6 |
| benign_suspicious_words | 8/10 (A7 F1 B2) | 9/10 (A9 F0 B1) | 8/10 (A8 F0 B2) | 9/10 |
| benign | 8/8 (A8 F0 B0) | 8/8 (A8 F0 B0) | 8/8 (A8 F0 B0) | 7/8 |

Cells show correct classifications / prompts, and the policy decisions (A=ALLOW, F=FLAG, B=BLOCK).
- Sentence-BERT + Logistic Regression: accuracy 0.9412, recall 1.0000, FPR 0.1111
- Sentence-BERT + Linear SVM: accuracy 0.9706, recall 1.0000, FPR 0.0556
- Sentence-BERT + Random Forest: accuracy 0.9412, recall 1.0000, FPR 0.1111
- TF-IDF + Logistic Regression (baseline): accuracy 0.9412, recall 1.0000, FPR 0.1111

### Challenge-set misclassifications (production model)

- [benign_suspicious_words] expected benign, got prompt_injection (BLOCK): "In Python, how do I ignore warnings from a specific library?"
- [benign_suspicious_words] expected benign, got prompt_injection (BLOCK): "Why do security teams say you should never reveal your password to anyone?"

## Train/test leakage check

- Exact duplicates after normalisation: {'train_val': 0, 'train_test': 0, 'val_test': 0}
- Highest cosine similarity of any test prompt to a training prompt: 0.796 (0 test prompts >= 0.9, 0 >= 0.8)
- Highest cosine similarity of any validation prompt to a training prompt: 0.970

## Production model selection

Selected: **Sentence-BERT + Logistic Regression** - validation composite 0.6*F1 + 0.4*(1-FPR); ties broken by lower validation Brier score.

- Sentence-BERT + Logistic Regression: composite 0.9621, Brier 0.0603
- Sentence-BERT + Random Forest: composite 0.9621, Brier 0.0943
- Sentence-BERT + Linear SVM: composite 0.9482, Brier 0.0256
