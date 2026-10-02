"""
PromptShield - Evaluation Module (standalone)
Run: python -m app.ml.evaluate

Re-evaluates the saved best model on the test set and generates reports.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")
SPLITS_DIR = os.path.join(BASE_DIR, "data", "splits")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
RESULTS_DIR = os.path.join(BASE_DIR, "evaluation", "results")


def main():
    print("=" * 60)
    print("PROMPTSHIELD - Model Evaluation")
    print("=" * 60)
    
    # Load model config
    with open(os.path.join(MODELS_DIR, "model_config.json")) as f:
        config = json.load(f)
    
    print(f"Model: {config['best_model_key']}")
    print(f"Representation: {config['representation']}")
    
    # Load model and data
    model = joblib.load(os.path.join(MODELS_DIR, "best_model.joblib"))
    le = joblib.load(os.path.join(MODELS_DIR, "label_encoder.joblib"))
    
    df_test = pd.read_csv(os.path.join(SPLITS_DIR, "test.csv"))
    y_test = le.transform(df_test['label'])
    
    # Load appropriate features
    if config['representation'] == 'minilm':
        X_test = np.load(os.path.join(PROCESSED_DIR, "test_embeddings.npy"))
    else:
        from app.ml.preprocess import preprocess_text
        tfidf = joblib.load(os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib"))
        test_texts = [preprocess_text(t) for t in df_test['prompt'].tolist()]
        X_test = tfidf.transform(test_texts)
    
    # Evaluate
    y_pred = model.predict(X_test)
    
    print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Per-category evaluation
    print("\nPer-Category Results:")
    print(f"{'Category':<30} {'Samples':>8} {'Correct':>8} {'Rate':>8}")
    print("-" * 60)
    
    for atype in df_test['attack_type'].unique():
        mask = df_test['attack_type'].values == atype
        y_sub = y_test[mask]
        y_pred_sub = y_pred[mask]
        correct = (y_sub == y_pred_sub).sum()
        rate = correct / len(y_sub) if len(y_sub) > 0 else 0
        print(f"{atype:<30} {len(y_sub):>8} {correct:>8} {rate:>8.4f}")
    
    print("\n✅ Evaluation complete.")


if __name__ == "__main__":
    main()
