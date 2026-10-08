#!/bin/bash
set -e

echo "=========================================================="
echo "RUNNING REPRODUCIBLE END-TO-END PIPELINE (final_model/)"
echo "=========================================================="

echo "[1/5] Running Data Audit and Leakage Assertions..."
python -m final_model.src.audit

echo "[2/5] Running Deterministic Labeller and Silver Label Generation..."
python -m final_model.src.labeller

echo "[3/5] Running Train Augmentation and Noisy Eval Sets Generation..."
python -m final_model.src.augment

echo "[4/5] Training Char-TFIDF Classifiers and Evaluating Baselines..."
python -m final_model.src.baselines

echo "[5/5] Running Final Frozen Evaluation on Test Split..."
python -m final_model.src.evaluate

echo "=========================================================="
echo "ALL STEPS COMPLETED SUCCESSFULLY!"
echo "Check final_model/outputs/predictions_test.csv and final_model/outputs/evaluation_metrics.json"
echo "=========================================================="
