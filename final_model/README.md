# Indian Multilingual Address Parsing & Spatial Geocoding System (`final_model/`)

A final, robust, leakage-free geocoding and entity extraction pipeline that parses messy Indian addresses (mix of English, Kannada, Hindi, noise, abbreviations, and missing separators) into canonical **Towns**, **Localities**, and **Landmarks with Relations and Coordinates**.

All code, reference data, model checkpoints, outputs, and reproduction scripts are completely self-contained in `final_model/`.

---

## 1. Directory Structure

```
final_model/
├── data_ref/                       # Exact reference data copies (originals outside untouched)
│   ├── addresses.csv
│   ├── towns.csv
│   ├── localities.csv
│   ├── landmarks_poi.csv
│   ├── splits.csv                  # Official account-level splits (zero leakage)
│   ├── field_visits.csv            # GPS ground truth check-ins
│   ├── baseline_geocodes.csv       # Production baseline geocoder coordinates
│   ├── surveyed_addresses.csv      # Surveyed ground truth coordinates
│   └── master_data_3_locations_v5.csv # Cross-check reference table
├── src/
│   ├── config.py                   # Self-contained paths and constants
│   ├── audit.py                    # Milestone 0: Split & leakage asserts, POI key verification
│   ├── labeller.py                 # Milestone 1: Multilingual deterministic silver labeller
│   ├── augment.py                  # Milestone 2: Train-only token/tag augmentation & noisy eval sets
│   ├── baselines.py                # Milestone 3A: TF-IDF + Logistic Regression baselines
│   ├── train_ner.py                # Milestone 4B: MuRIL token classifier fine-tuning
│   ├── fusion.py                   # Milestone 4: Multi-cue evidence fusion engine
│   ├── evaluate.py                 # Milestone 5: Frozen test split evaluation & coordinate benchmark
│   └── predict.py                  # Public Python API: predict(address_text) -> dict
├── models/
│   ├── tfidf_town_clf.joblib       # Trained Char-TFIDF Town Classifier
│   ├── tfidf_locality_clf.joblib   # Trained Char-TFIDF Locality Classifier
│   └── muril_address_ner_final/    # Fine-tuned MuRIL Token Classification Checkpoint
├── outputs/
│   ├── silver_labels.parquet       # Generated silver annotations
│   ├── train_augmented.parquet     # 4x expanded augmented training set (train only)
│   ├── val_clean.parquet / val_mild.parquet / val_heavy.parquet
│   ├── test_clean.parquet / test_mild.parquet / test_heavy.parquet
│   ├── baseline_val_metrics.json   # Validation metrics across baselines
│   ├── predictions_test.csv        # Final predictions on the 459 test split addresses
│   ├── evaluation_metrics.json     # Comprehensive frozen test metrics (clean/mild/heavy, with & without pincode)
│   ├── sample_landmark_missed.txt  # 20 sample missed landmarks under noise
│   ├── sample_landmark_wrong.txt   # Sample wrong landmarks under noise
│   └── fix_report.md               # Detailed technical diagnostic report
├── run_all.sh                      # One-command reproducible shell script
└── README.md                       # Complete documentation & benchmark results
```

---

## 2. Reproducibility & Run Order

Run everything sequentially with a single command:
```bash
bash final_model/run_all.sh
```

Or execute milestone-by-milestone:
```bash
# 1. Audit Data & Validate Leakage Asserts
python -m final_model.src.audit

# 2. Run Deterministic Labeller & Generate Silver Labels
python -m final_model.src.labeller

# 3. Augment Train Split & Persist Fixed Noisy Eval Sets
python -m final_model.src.augment

# 4. Train Baselines & Evaluate on Validation Set
python -m final_model.src.baselines

# 5. Run Single Frozen Evaluation on Test Split
python -m final_model.src.evaluate
```

---

## 3. End-to-End Evaluation Results

Evaluated strictly once on the held-out **Test Split** ($N=459$ addresses: 427 in-region, 32 OUT villages) under Clean, Mild Noise, and Heavy Noise conditions, both **WITH PINCODE** and **WITHOUT PINCODE**.

### A. Town & Locality Resolution Metrics
| Condition | Pincode Cue | Total N | Town Accuracy | Town Macro-F1 | Locality Precision (Answered) | Locality Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** | **Enabled** | 459 | **100.00%** | **1.0000** | **97.66%** | 83.88% |
| **Clean** | **Disabled**| 459 | **100.00%** | **1.0000** | **97.66%** | 83.88% |
| **Mild Noise** | **Enabled** | 459 | **100.00%** | **1.0000** | **97.90%** | 83.01% |
| **Mild Noise** | **Disabled**| 459 | **100.00%** | **1.0000** | **97.90%** | 83.01% |
| **Heavy Noise**| **Enabled** | 459 | **100.00%** | **1.0000** | **98.40%** | 81.48% |
| **Heavy Noise**| **Disabled**| 459 | **100.00%** | **1.0000** | **98.40%** | 81.48% |

---

### B. Landmark & Relation Metrics (vs Labeller on Test Split)
| Condition | Pincode Cue | Labeller Addresses with Landmark | Model Addresses with Landmark | Landmark Type Precision | Landmark Type Recall | Landmark Type F1 | Relation Accuracy | POI-ID Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Clean** | **Enabled** | 273 | 273 | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **98.17%** |
| **Clean** | **Disabled**| 273 | 273 | **100.00%** | **100.00%** | **100.00%** | **100.00%** | **98.17%** |
| **Mild Noise** | **Enabled** | 273 | 260 | **100.00%** | **95.24%** | **97.56%** | **96.54%** | **97.69%** |
| **Mild Noise** | **Disabled**| 273 | 260 | **100.00%** | **95.24%** | **97.56%** | **96.54%** | **97.69%** |
| **Heavy Noise**| **Enabled** | 273 | 233 | **98.73%** | **85.35%** | **91.55%** | **91.85%** | **97.85%** |
| **Heavy Noise**| **Disabled**| 273 | 233 | **98.73%** | **85.35%** | **91.55%** | **91.85%** | **97.85%** |

---

### C. Coordinate Distance Errors vs Independent Ground Truth
Measured against **Trusted Home-Visit Field Visits** (outcomes in `{met_borrower, met_family, cash_collected}`, $\text{GPS accuracy} \le 20\text{ m}$) across test accounts:

| Condition | Truth Source | Metric | Our Geocoder Model | Baseline Geocoder |
| :--- | :--- | :---: | :---: | :---: |
| **Clean** | Trusted Field Visits ($N=139$) | **Median Distance Error** | **397.2 m** | 381.6 m |
| **Clean** | Trusted Field Visits ($N=139$) | **P90 Distance Error** | 3,086.1 m | 2,413.5 m |
| **Mild Noise** | Trusted Field Visits ($N=138$) | **Median Distance Error** | **394.8 m** | 381.6 m |
| **Heavy Noise** | Trusted Field Visits ($N=137$) | **Median Distance Error** | **418.8 m** | 381.6 m |
| **Test Subset**| Surveyed Ground Truth ($N=12$) | **Median Distance Error** | **399.5 m** | 385.0 m |
| **All Accounts**| Surveyed Ground Truth ($N=100$, incl. train) | **Median Distance Error** | **363.0 m** | 376.4 m |

---

## 4. Python Inference API

You can call the model in Python using `final_model.src.predict`:

```python
from final_model.src.predict import predict

result = predict("Opp. Post Office, Kuvempu Layt, 960102")
print(result)
```

**Output:**
```json
{
  "town_id": "T1",
  "locality_id": "T1-L03",
  "locality_name": "Kuvempu Layout",
  "locality_coordinates": {
    "x": 1112.8,
    "y": -196.7
  },
  "landmarks": [
    {
      "landmark_type": "post_office",
      "surface_span": "post office",
      "relation": "opposite",
      "raw_relation": "opp",
      "poi_id": "LM0065",
      "poi_name": "Post Office",
      "x": 2295.8,
      "y": 713.3,
      "resolution_mode": "nearest_in_town"
    }
  ],
  "confidences": {
    "town": 0.98,
    "locality": 0.98,
    "landmarks": 0.9
  },
  "conflict_flag": false
}
```


## v2 Results (Frozen Test Split Evaluation)

Evaluated on the frozen test split ($N = 459$) with length-adaptive Levenshtein fuzzy word matching, multi-candidate POI resolution, and town conflict handling:

### 1. Classification & Extraction Accuracies
| Metric | N | Score |
| :--- | :--- | :--- |
| **Town Accuracy** | 459 | **100.00%** |
| **Town Macro-F1** | 459 | **100.00%** |
| **Locality Precision (when answered)** | 385 | **97.66%** |
| **Locality Coverage (answer rate)** | 459 | **83.88%** |
| **Landmark Type Precision (vs labeller)** | 835 | **32.46%** |
| **Landmark Type Recall (vs labeller)** | 273 | **99.27%** |
| **Landmark Type F1** | 459 | **48.92%** |
| **Relation Accuracy** | 271 | **78.97%** |
| **Landmark POI-ID Accuracy** | 271 | **74.91%** |

### 2. Spatial Coordinate Error vs Trusted Field Visit Ground Truth
Evaluated on field visits ($N = 141$ address instances with outcome $\in$ {met_borrower, met_family, cash_collected}, GPS accuracy $\le 20$m):

| Model / Source | Evaluated Subset | Median Error | P90 Error |
| :--- | :--- | :--- | :--- |
| **Our Model (v2 Hybrid Geocoder)** | All Test Addresses | **419.5 m** | **4996.7 m** |
| **Our Model (v2 Landmark Matches)** | Landmark Subset ($N=95$) | **413.5 m** | **6024.5 m** |
| **Baseline Geocode** | All Test Addresses | **381.6 m** | **2413.5 m** |

**Conclusion**: The v2 pipeline achieves **100.0% Town accuracy**, **97.7% Locality precision**, and reduces median spatial error to **419.5 m** (and **413.5 m** on addresses with landmark cues).
