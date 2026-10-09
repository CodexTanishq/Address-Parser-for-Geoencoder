# OpenIIT 2.0 address parser architecture

## 1. Goal

The project turns Indian address text into structured address information and coordinates.
It handles English, Hindi written in Devanagari, and Kannada, including mixed-script lines.
The final flow predicts town, locality, landmarks, and a coordinate candidate without retraining at inference time.

## 2. Data

All row counts below are physical CSV data rows. The descriptions are based on the column names and the scripts that read them.

| File | Rows | Columns | What it is |
|---|---:|---|---|
| `data/addresses.csv` | 3,117 | `address_id, account_id, address_type, source, added_date, town_id, address_text` | Raw source addresses and their account and town IDs. |
| `data/splits.csv` | 2,400 | `account_id, split` | Account-level train, validation, and test assignment. |
| `data/towns.csv` | 3 | `town_id, town_name, address_style, approx_radius_m` | Town reference table. It has no coordinate columns. |
| `data/localities.csv` | 36 | `locality_id, town_id, locality_name, pincode, centroid_x, centroid_y` | Locality reference table with locality centroids and pincodes. |
| `data/landmarks_poi.csv` | 240 | `poi_id, town_id, landmark_type, name, x, y` | Landmark POIs and their coordinates. There are 14 landmark types. |
| `data/baseline_geocodes.csv` | 2,880 | `address_id, geocoder_x, geocoder_y, precision` | External baseline geocode point and its precision label. It is used to choose among candidate POIs, not as true coordinates. |
| `data/field_visits.csv` | 5,578 | `visit_id, account_id, address_id, agent_id, visit_date, start_ts, checkin_ts, checkin_x, checkin_y, gps_accuracy_m, dwell_s, outcome, ptp_id, remark, photo_hash` | Field-visit records. It is not in the final inference scripts inspected. |
| `data/visit_gps_points.csv` | 160,406 | `visit_id, seq, point_ts, x, y, accuracy_m` | GPS point traces for field visits. Not in the final inference scripts inspected. |
| `data/surveyed_addresses.csv` | 100 | `address_id, surveyed_x, surveyed_y` | Small surveyed-coordinate table. Not used by the final coordinate script. |
| `data/stress_set.csv` | 100 | `address_id, town_id, address_text, matched_groups, landmark_has_relation, stressed_address_text` | Train-only typo/spelling stress set made by script 01. |
| `data/test_noout_ids.csv` | 427 | `address_id` | Test IDs whose source `town_id` is not `OUT`. |
| `data/cleaned/train_cleaned.csv` | 2,175 | `address_id, town_id, address_text, matched_groups, landmark_has_relation` | Transliteration and word-group result for train. |
| `data/cleaned/validation_cleaned.csv` | 483 | same as train cleaned | Transliteration and word-group result for validation. |
| `data/cleaned/test_cleaned.csv` | 459 | `address_id, town_id, address_text, matched_groups` | Existing cleaned test file. It was not read by the final coordinate script. |
| `data/cleaned/unknown_words.csv` | 395 | `word, count` | Unmatched non-filler words from preparation. |
| `data/labels/train_bio.csv` | 24,591 word rows | `address_id, town_id, word_index, word, tag, landmark_type, pincode_is_real, label_quality` | Automatic BIO labels before AI-label merge. |
| `data/labels/train_bio_merged.csv` | 23,481 word rows | same BIO columns | First accepted-AI-label merge, excluding hold-out IDs. |
| `data/labels/train_bio_merged_v2.csv` | 23,481 word rows | same BIO columns | Version 2 merge that includes the recovered batch 001. |
| `data/labels/train_bio_merged_v2_noout.csv` | 22,379 word rows | same BIO columns | Version 2 training labels with `town_id=OUT` addresses removed. |
| `data/labels/validation_bio.csv` | 5,416 word rows | same BIO columns | Validation BIO labels including `OUT`. |
| `data/labels/validation_bio_noout.csv` | 5,134 word rows | same BIO columns | Validation BIO labels excluding `OUT`. |
| `data/labels/review_sample.csv` | 150 | `address_id, town_id, label_quality, labelled_address` | Human-review sample produced by script 02. |

### Label fields

`tag` is one of BIO tags for `TOWN`, `LOCALITY`, `LANDMARK`, `RELATION`, `PINCODE`, or `O`. `landmark_type` is populated on landmark-tagged tokens by the automatic labeler. It is not independently surveyed truth, so landmark-type metrics against it are circular. `label_quality` comes from the locality labelling rule: `sure`, `fuzzy`, `no_locality`, or `conflict`.

## 3. Pipeline scripts

Commands assume the current folder is the project root and Python is `python`. The recorded environment uses `C:\Users\bajor\AppData\Local\Programs\Python\Python312\python.exe`.

| Script | Reads | Does | Writes | Command |
|---|---|---|---|---|
| `01_prepare_data.py` | `addresses.csv`, `splits.csv` | Checks account/address split leakage; transliterates Hindi and Kannada; normalizes text; builds train-derived fuzzy seed groups; creates matched-group and relation proximity fields. | `data/cleaned/train_cleaned.csv`, `validation_cleaned.csv`, `unknown_words.csv`, `data/stress_set.csv` | `python code_files/01_prepare_data.py` |
| `02_auto_label.py` | Cleaned train/validation, towns, localities, POIs | Creates conservative word-level BIO labels. Uses town-aware locality matching and rule-based landmark/relation labels. | `data/labels/train_bio.csv`, `validation_bio.csv`, `review_sample.csv` | `python code_files/02_auto_label.py` |
| `03_build_train_merged.py` | `train_bio.csv`, hold-out IDs, accepted/rejected AI label gates | Uses accepted AI tags, fallback tags for rejects, fixes orphan `I-` tags, removes hold-out IDs, validates word identity and BIO. | `data/labels/train_bio_merged.csv` | `python code_files/03_build_train_merged.py` |
| `03b_build_train_merged_v2.py` | `train_bio.csv`, batch 001, AI changes and gate files | Restores batch 001, applies its cut-short fix, runs the label gate, builds v2 merged labels, and treats specified invalid-`I` gate failures carefully. | `out_batch001.csv`, `out2_batch001.csv`, `check2_batch001/`, `train_bio_merged_v2.csv` | `python code_files/03b_build_train_merged_v2.py` |
| `04_train_ner.py` | Merged train labels and validation labels | Trains MuRIL token classification; uses first-subword labels, typo noise on train only, and masks PINCODE in loss. | Training work directory and a model directory configured by wrapper scripts. | `python code_files/04_train_ner.py` |
| `04b_train_ner_v2.py` | Same as 04, using `train_bio_merged_v2.csv` | Wrapper that changes v1 locations to v2 locations. | `temp_code/ner_training_v2`, intended `models/ner_v2` | `python code_files/04b_train_ner_v2.py` |
| `04c_train_ner_v3.py` | No-OUT train and validation labels | Wrapper for no-OUT v3 training. Stops if `models/ner_v3_best` exists. | `temp_code/ner_training_v3`, `models/ner_v3_best` | `python code_files/04c_train_ner_v3.py` |
| `05_infer.py` | Validation BIO, addresses, towns, localities, POIs, NER model | Early validation inference and town/locality/POI matching. | `outputs/val_predictions.csv` if run in its original state. | `python code_files/05_infer.py` |
| `05b_infer_v2.py` | Validation labels, addresses, splits, merged v2 labels, reference tables | V2 validation inference and early matching diagnostics. | `outputs/val_predictions_v2.csv`, `val_errors_v2.csv`, `report_v2.txt` | `python code_files/05b_infer_v2.py` |
| `05c_infer_v3.py` | No-OUT validation, addresses, reference tables | V3 validation-only town/locality/landmark inference, with pincode and WRatio logic. | `outputs/val_predictions_v3.csv`, `val_errors_v3.csv`, `report_v3.txt` | `python code_files/05c_infer_v3.py` |
| `06_final_pipeline.py` | No-OUT validation/train labels, addresses, towns, localities, POIs | Validation-only final-pipeline prototype. Trains a small character landmark classifier on train labels and assigns POI catchments. | `val_predictions_final_v2.csv`, errors and reports. | `python code_files/06_final_pipeline.py` |
| `07_holdout_check.py` | No-OUT hold-out IDs, addresses, no-OUT train labels, reference tables | Runs a fixed pre-11 pipeline on 90 hold-out rows. | `outputs/holdout_predictions.csv` | `python code_files/07_holdout_check.py` |
| `08_ner_output.py` | Train/no-OUT validation labels, addresses, reference tables, IDs argument | Produces NER-output tables and NER metrics. | `ner_output_validation.csv`, `ner_output_holdout.csv`, metrics/report files. | `python code_files/08_ner_output.py --ids data/labels/validation_bio_noout.csv` |
| `08b_ner_output.py` | Same main inputs plus no-OUT hold-out IDs | Fixes town fallback/locality IDs and produces v2 NER output. | `ner_output_validation_v2.csv`, `ner_output_holdout_v2.csv`, `ner_metrics_v2.csv`, report. | `python code_files/08b_ner_output.py` |
| `09_kfold_check.py` | No-OUT merged train labels, no-OUT hold-out IDs, addresses/towns | Five-fold GroupKFold check by account ID, using the 04 settings. Fold models are temporary. | `kfold_metrics.csv`, `kfold_report.txt` | `python code_files/09_kfold_check.py` |
| `10_final_test.py` | `test_noout_ids.csv`, raw addresses, no-OUT merged train labels, reference tables | Makes test NER output with `ner_v2_best`; trains the small landmark classifier on train labels. | `test_ner_output.csv`, `test_report.txt` | `python code_files/10_final_test.py` |
| `11_coordinates.py` | An NER-output CSV, baseline geocodes, no-OUT merged train labels, localities, POIs | Final coordinate selection. Uses baseline geocode only to choose among candidate POIs. | `<prefix>.csv`, `<prefix>_low_probability_spans.csv`, `<prefix>_report.txt` | `python code_files/11_coordinates.py outputs/test_ner_output.csv outputs/coordinates_test` |

`code_files/__pycache__/` contains Python bytecode for scripts 01, 02, 03, 04, 05, and 11. It is derived cache, not a pipeline input.

## 4. Models

### NER models

| Folder | Base model / task | Training data | Settings visible in source | Training script |
|---|---|---|---|---|
| `models/ner_v2_best/` | Google `muril-base-cased`, BERT token classification for ten BIO labels | V2 merged train labels. The trained checkpoint exported from `temp_code/ner_training_v2/checkpoint-520`; exact final export command is not stored in a script. | 5 epochs, learning rate `3e-5`, batch 16, fp16, seed 42, maximum token length 128, first-subword labels; retry is batch 8 with accumulation 2 on CUDA OOM. | `04_train_ner.py` via `04b_train_ner_v2.py` |
| `models/ner_v3_best/` | Same MuRIL token classification architecture | `train_bio_merged_v2_noout.csv`; no-OUT validation. | Same source settings as above. `training_args.bin` is present. | `04_train_ner.py` via `04c_train_ner_v3.py` |

Both exported configs declare the same labels in this order: `O`, `B-TOWN`, `I-TOWN`, `B-LANDMARK`, `I-LANDMARK`, `B-LOCALITY`, `I-LOCALITY`, `B-RELATION`, `I-RELATION`, `B-PINCODE`. Each model folder has config, tokenizer files, and a 947,916,648-byte `model.safetensors` file. The raw tensor contents were not read.

### Landmark-type classifier

There is no saved classifier folder. Scripts 06, 07, 08, 08b, 10, and 11 rebuild it in memory from landmark spans in `train_bio_merged_v2_noout.csv` using character TF-IDF n-grams `(2, 5)` and `LogisticRegression(max_iter=1000, random_state=42)`. It is trained from train labels only.

### Qwen in final coordinates

The final script 11 uses `Qwen/Qwen2.5-0.5B-Instruct` from the Hugging Face cache at `C:\Users\bajor\.cache\huggingface\hub\models--Qwen--Qwen2.5-0.5B-Instruct`. The cache was measured at about 0.93 GB. It scores the 14 allowed landmark types by log-likelihood. For low classifier confidence, script 11 subtracts each type score for the content-free span `N/A`, normalizes the calibrated scores, and averages those probabilities equally with classifier probabilities. It is not trained by this project.

## 5. Libraries and environment

Python is `3.12.10`.

| Package | Version | Used for |
|---|---:|---|
| `torch` | `2.5.1+cu121` | NER and Qwen inference. |
| `transformers` | `5.15.0` | MuRIL, Qwen, Trainer, tokenizers. |
| `pandas` | `3.0.5` | CSV reading/writing and tables. |
| `numpy` | `2.5.2` | Present in the temporary comparison script only. |
| `scikit-learn` | `1.9.0` | TF-IDF, logistic regression, GroupKFold. |
| `seqeval` | `1.2.2` | BIO entity metrics. |
| `RapidFuzz` | `3.14.6` | Fuzzy town/locality matching. |
| `indic_transliteration` | `2.3.82` | Hindi/Kannada transliteration. |
| `huggingface_hub` | `1.27.0` | Model download/cache support. |

`torchvision` is not installed in the current Python environment. The exact earlier torchvision error and repair steps are not recorded in the project files, so not sure. None of the final scripts imports torchvision. `sentence-transformers` is also not installed; the temporary experiment used raw Transformers mean pooling instead.

## 6. Rules in the final coordinate flow

The final flow is `08b_ner_output.py` followed by `11_coordinates.py`.

1. **Town:** NER predicts a town span. Five- or six-digit tokens are ignored in spans. Town matching uses WRatio with normalized processing and a cutoff of 85. If no town is found, a real pincode may choose the town when that pincode maps to one town.
2. **Locality:** The NER locality span is fuzzy-matched only among localities in the predicted town. A pincode helps only where the applicable script allows it; an NER locality wins when present.
3. **Landmark type:** A train-only character TF-IDF/logistic classifier predicts a type. If classifier probability is at least 0.5, the classifier type is kept. Below 0.5, script 11 uses an equal average of classifier probabilities and content-free-calibrated Qwen probabilities. The resulting type source is `avg_llm`.
4. **Candidate levels:** Script 11 stops at the first level with at least one POI of the final type: (1) nearest-locality catchment in the town, (2) other localities with the same pincode in the town, (3) pincode localities when locality is unknown, then (4) all POIs of that type in the predicted town.
5. **POI choice:** From those candidates, select the POI nearest in Euclidean distance to `baseline_geocodes.csv`'s geocoder point. The choice remains inside the predicted town.
6. **No-landmark area fallback:** Use locality centroid when locality is known; otherwise pincode-area centroid when the pincode resolves to localities; otherwise mean centroid of all localities in the town. A blank town becomes `unknown_town` and has no safe centroid.

The current test-coordinate result has 427 rows, 427 unique address IDs, zero absent baseline IDs, and zero zero-candidate landmark rows. Coordinate quality cannot be scored because no true coordinate target was used for that evaluation.

## 7. `temp_code` contents

`temp_code` is intentionally outside the correct-flow directory. It contains:

- `labeller_batches/` (52 train-only CSV batches), `labeller_compact/` (their compact text form), and `labeller_big/` (14 larger batch CSV/text pairs) for human/AI labelling.
- `labeller_out/` with changes files, merged outputs, accepted/rejected gate outputs, and cut-short correction artifacts. These were used to build merged labels.
- `check_ai_labels.py`, `merge_changes.py`, `make_compact_batches.py`, `find_cutshort.py`, `make_cutshort_changes.py`, and `show_samples.py`: supporting labelling and review tools.
- `holdout_ids.csv` and `holdout_ids_noout.csv`: account-grouped hold-out IDs.
- `11_coordinates.py` and `low_prob_spans_to_label.csv`: experimental coordinate/type-comparison code and manually supplied evaluation labels. They are not read by final script 11.
- CSV samples, candidate lists, gate summaries, logs, and Python bytecode caches. These are audit or temporary artifacts, not final-flow inputs.

## 8. Known weak points

- Landmark types are weak for some transliterated or misspelled words. The temporary comparison found that the equal classifier/Qwen average worked well on the 55 manually labelled low-probability occurrences, but this is a small evaluation set.
- `landmark_type` in automatic BIO labels comes from script 02. Any score against it is circular rather than independent ground truth.
- There is no true locality field in `addresses.csv`, and the coordinate reports state that true coordinates are unavailable. Coordinate quality therefore cannot be measured from the final outputs alone.
- Baseline geocoder points are used to choose between POIs. Their `precision` values include `locality`, `street`, `pincode`, and `rooftop`, so they are rough for some rows and may bias the selected POI.
- One validation coordinate row has `unknown_town`; this is reported by the final validation coordinate run.
- The source project has several older outputs and prototype scripts. The correct coordinate flow is the final `code_files/11_coordinates.py`, not the temporary comparison script.
- `data/cleaned/test_cleaned.csv` exists, even though the later final run used `test_noout_ids.csv` and raw text. Its provenance relative to the latest final flow is not sure.
