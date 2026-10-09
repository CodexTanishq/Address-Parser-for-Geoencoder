# OpenIIT 2.0 â€” Results

All values below are percentages rounded to two decimal places unless stated otherwise. No test labels were read or used.

## How these results were made

- Final NER model: `models/ner_v2_best`.
- Landmark-type classifier: rebuilt from train data only, as in `code_files/08b_ner_output.py`.
- Final landmark-type rule: retain the classifier type at probability at least 0.50; otherwise average its distribution with the content-free calibrated Qwen2.5-0.5B constrained-choice distribution.
- `temp_code/final_scores.py` reproduces the calculations. It uses the existing `outputs/ner_metrics.csv` values for the published train and validation BIO report, as requested. It measures the 90-address hold-out directly.
- The training split was seen by the model during fitting, so its scores are in-sample diagnostics, not independent generalisation results.

## Train (1,920 non-OUT addresses; seen during training)

NER entity-level BIO scores from the published `outputs/ner_metrics.csv`:

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| Overall | 97.54% | 98.68% | 98.11% |
| TOWN | 98.91% | 98.96% | 98.93% |
| LOCALITY | 95.82% | 100.00% | 97.86% |
| LANDMARK | 95.39% | 95.45% | 95.42% |
| RELATION | 100.00% | 100.00% | 100.00% |
| PINCODE | 0.00% | 0.00% | 0.00% |

Final structured-output comparison:

| Measure | Score |
|---|---:|
| town_id accuracy | 99.95% |
| locality_id accuracy, including matching blanks | 96.41% |
| landmark count accuracy | 99.53% |
| landmark span exact-match accuracy | 96.61% |

## Validation (443 non-OUT addresses)

NER entity-level BIO scores from the published `outputs/ner_metrics.csv`:

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| Overall | 79.49% | 93.18% | 85.79% |
| TOWN | 99.55% | 99.55% | 99.55% |
| LOCALITY | 97.34% | 100.00% | 98.65% |
| LANDMARK | 34.46% | 56.04% | 42.68% |
| RELATION | 69.42% | 99.02% | 81.62% |
| PINCODE | 0.00% | 0.00% | 0.00% |

Final structured-output comparison, using `outputs/ner_output_validation_v2.csv`:

| Measure | Score |
|---|---:|
| town_id accuracy | 99.77% |
| locality_id accuracy, including matching blanks | 97.97% |
| landmark count accuracy | 74.04% |
| landmark span exact-match accuracy | 56.21% |

## Hold-out (90 addresses from the original train split)

These addresses were excluded from the merged NER training data. BIO labels come from `data/labels/train_bio.csv`.

| Entity | Precision | Recall | F1 |
|---|---:|---:|---:|
| Overall | 48.94% | 42.86% | 45.70% |
| TOWN | 4.44% | 4.44% | 4.44% |
| LOCALITY | 95.06% | 100.00% | 97.47% |
| LANDMARK | 41.07% | 67.65% | 51.11% |
| RELATION | 61.82% | 97.14% | 75.56% |
| PINCODE | 0.00% | 0.00% | 0.00% |

Final structured-output comparison, using `outputs/ner_output_holdout_v2.csv`:

| Measure | Score |
|---|---:|
| town_id accuracy | 100.00% |
| locality_id accuracy, including matching blanks | 95.56% |
| landmark count accuracy | 75.56% |
| landmark span exact-match accuracy | 63.33% |

## Landmark-type check on the 55 hand-labelled low-confidence spans

The 55 occurrences are the validation and hold-out low-probability landmark spans labelled by hand in `temp_code/low_prob_spans_to_label.csv`. They were used only for this measurement, never for training or for a spelling/rule list.

| Method | Correct / 55 | Accuracy |
|---|---:|---:|
| Landmark classifier alone | 31 / 55 | 56.36% |
| Final average rule: classifier + calibrated Qwen | 45 / 55 | 81.82% |

For all other landmark-type labels, any score against the automatically produced labels from `02_auto_label.py` is **circular** and is only a reference. The direct classifier is 100.00% on its train examples, which is in-sample and not a useful generalisation estimate. The earlier full NER-plus-classifier circular reference in `outputs/ner_metrics.csv` is 68.65% on train and 72.46% on validation.

## Five-fold grouped check

The grouped folds use train labels only and exclude the 90 hold-out rows. The final deployed model remains `ner_v2_best`; no fold model is the final model.

| Metric | Mean | Standard deviation |
|---|---:|---:|
| Token accuracy | 97.78% | 0.27 pp |
| Token accuracy without O | 97.90% | 0.29 pp |
| Entity precision without PINCODE | 92.32% | 0.70 pp |
| Entity recall without PINCODE | 95.92% | 0.42 pp |
| Entity F1 without PINCODE | 94.08% | 0.54 pp |
| TOWN F1 | 98.93% | 0.44 pp |
| LOCALITY F1 | 97.95% | 0.31 pp |
| LANDMARK F1 | 94.87% | 1.82 pp |
| RELATION F1 | 82.47% | 2.99 pp |
| town_id accuracy | 99.90% | 0.14 pp |
| locality exact-match accuracy | 96.51% | 0.39 pp |

## Test

no true labels; only the user's hand check of 125 of 427 rows: town 100%, locality 100%, landmark present/absent 100%, landmark text 100%, landmark type about 80-85% before the avg rule (not re-measured)

## What these numbers mean â€” and do not mean

Town and locality resolution are strong on the validation and hold-out data after the final resolver rules. Landmark extraction is weaker, especially on validation, so coordinate choices based on landmarks need more caution than town or locality choices. PINCODE has zero NER score because it is intentionally handled by a rule rather than trained in the NER loss.

The training scores are not independent evidence because the model trained on those addresses. The hold-out result is more useful for an independent check, but it has only 90 rows. The hand-labelled low-confidence landmark set is the only non-circular landmark-type evaluation here; it is small and focused on difficult cases. Coordinates cannot be scored because there are no true coordinates.

## Notes on two low numbers

- Hold-out TOWN F1 (4.44%) is a raw BIO-tag score. The final town_id is 100% correct because the output step falls back to the pincode when the NER town span is weak. The two numbers measure different stages. Use town_id accuracy as the town result.
- Validation and hold-out LANDMARK precision is low because the automatic gold labels (02_auto_label.py) often miss a landmark or cut it short (gold 'mantapa' vs predicted 'kalyana mantapa'; gold 'none' vs predicted 'hanuman mandir'). In 25 checked validation examples all predicted spans looked like real landmarks. Landmark precision against these labels is therefore too low, and a hand check on test (125 rows) found landmark text 100% correct.
- Test has no true labels, so the only test score is the 125-row hand check. Coordinates cannot be scored, because there are no true coordinates. Train scores are in-sample.
