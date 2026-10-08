"""
Milestone 3A: Fast Machine Learning Baselines
- Evaluates on the exact same splits and fixed noisy sets (clean, mild, heavy):
  Baseline 1: Deterministic Labeller itself
  Baseline 2: Gazetteer Linker (Fuzzy N-gram + rapidfuzz)
  Baseline 3: Character TF-IDF (1-5 grams) + Logistic Regression (Town & Locality)
"""
import sys
import io
import time
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from final_model.src import config
from final_model.src.labeller import DeterministicLabeller

def train_and_eval_tfidf_baselines():
    print("=" * 70)
    print("TRAINING CHAR-TFIDF + LOGISTIC REGRESSION BASELINES")
    print("=" * 70)
    
    # 1. Load Augmented Train Set
    train_df = pd.read_parquet(config.OUTPUTS_DIR / "train_augmented.parquet")
    print(f"Loaded {len(train_df)} training samples.")
    
    # Train Town Classifier
    print("Training Char-TFIDF Town Classifier...")
    town_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2)),
        ('clf', LogisticRegression(C=5.0, max_iter=500, random_state=config.RANDOM_SEED))
    ])
    town_pipe.fit(train_df['text'], train_df['gold_town_id'])
    joblib.dump(town_pipe, config.TFIDF_TOWN_MODEL_PATH)
    print(f"  [OK] Saved Town Model to {config.TFIDF_TOWN_MODEL_PATH}")
    
    # Train Locality Classifier (Filter for known localities)
    loc_train = train_df[train_df['silver_locality_id'] != 'UNKNOWN']
    print(f"Training Char-TFIDF Locality Classifier on {len(loc_train)} samples with locality...")
    loc_pipe = Pipeline([
        ('tfidf', TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5), min_df=2)),
        ('clf', LogisticRegression(C=5.0, max_iter=500, random_state=config.RANDOM_SEED))
    ])
    loc_pipe.fit(loc_train['text'], loc_train['silver_locality_id'])
    joblib.dump(loc_pipe, config.TFIDF_LOCALITY_MODEL_PATH)
    print(f"  [OK] Saved Locality Model to {config.TFIDF_LOCALITY_MODEL_PATH}")
    
    return town_pipe, loc_pipe

def evaluate_all_baselines():
    start_time = time.time()
    print("=" * 70)
    print("EVALUATING BASELINES ON VALIDATION SET (CLEAN, MILD, HEAVY)")
    print("=" * 70)
    
    town_pipe, loc_pipe = train_and_eval_tfidf_baselines()
    labeller = DeterministicLabeller()
    
    eval_results = {}
    
    for condition in ['clean', 'mild', 'heavy']:
        val_df = pd.read_parquet(config.OUTPUTS_DIR / f"val_{condition}.parquet")
        n = len(val_df)
        print(f"\n--- Condition: {condition.upper()} (N={n}) ---")
        
        # 1. Deterministic Labeller / Linker
        labeller_preds = [labeller.label_address(txt) for txt in val_df['text']]
        lab_towns = [p['town_id'] for p in labeller_preds]
        lab_locs = [p['locality_id'] for p in labeller_preds]
        
        # 2. TF-IDF + Logistic Regression
        tfidf_towns = town_pipe.predict(val_df['text'])
        # Locality probs + thresholding
        tfidf_loc_probs = loc_pipe.predict_proba(val_df['text'])
        max_probs = np.max(tfidf_loc_probs, axis=1)
        pred_loc_idx = np.argmax(tfidf_loc_probs, axis=1)
        classes = loc_pipe.classes_
        tfidf_locs = [classes[idx] if prob > 0.45 else 'UNKNOWN' for idx, prob in zip(pred_loc_idx, max_probs)]
        
        # Gold targets
        gold_towns = val_df['gold_town_id'].tolist()
        silver_locs = val_df['silver_locality_id'].tolist()
        
        # Accuracies
        acc_town_lab = accuracy_score(gold_towns, lab_towns)
        acc_town_tfidf = accuracy_score(gold_towns, tfidf_towns)
        
        # Locality accuracy when gold is known
        loc_mask = [g != 'UNKNOWN' for g in silver_locs]
        acc_loc_lab = np.mean([p == g for p, g, m in zip(lab_locs, silver_locs, loc_mask) if m])
        acc_loc_tfidf = np.mean([p == g for p, g, m in zip(tfidf_locs, silver_locs, loc_mask) if m])
        
        # Locality Coverage (Answer rate)
        cov_loc_lab = np.mean([p != 'UNKNOWN' for p in lab_locs])
        cov_loc_tfidf = np.mean([p != 'UNKNOWN' for p in tfidf_locs])
        
        print(f"  [Labeller/Linker] Town Acc: {acc_town_lab*100:.2f}% | Locality Acc (Answered): {acc_loc_lab*100:.2f}% | Loc Cov: {cov_loc_lab*100:.2f}%")
        print(f"  [TF-IDF + LogReg] Town Acc: {acc_town_tfidf*100:.2f}% | Locality Acc (Answered): {acc_loc_tfidf*100:.2f}% | Loc Cov: {cov_loc_tfidf*100:.2f}%")
        
        eval_results[condition] = {
            'N': n,
            'labeller': {
                'town_acc': round(acc_town_lab, 4),
                'locality_acc': round(acc_loc_lab, 4),
                'locality_coverage': round(cov_loc_lab, 4)
            },
            'tfidf_logreg': {
                'town_acc': round(acc_town_tfidf, 4),
                'locality_acc': round(acc_loc_tfidf, 4),
                'locality_coverage': round(cov_loc_tfidf, 4)
            }
        }
        
    out_file = config.OUTPUTS_DIR / "baseline_val_metrics.json"
    with open(out_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n[OK] Baseline validation metrics saved to {out_file}")
    
    elapsed = time.time() - start_time
    print(f"\n[Baselines Elapsed Time: {elapsed:.2f}s]")
    return eval_results

if __name__ == "__main__":
    evaluate_all_baselines()
