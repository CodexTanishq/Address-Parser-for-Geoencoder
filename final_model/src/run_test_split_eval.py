"""
Frozen Test Split Evaluation for Final Model v2.
Runs ONCE on the frozen test split (splits.csv, split = test).
Computes:
- Town Accuracy & Macro-F1 (N=459)
- Locality Precision (when answered) & Coverage (N=459)
- Landmark Type Precision, Recall, F1 (vs labeller)
- Landmark Relation Accuracy & POI-ID Accuracy
- Landmark & Locality Coordinate Error (Median and P90) vs Trusted Field Visit Truth
Updates final_model/README.md with a "v2 Results" section.
"""
import io
import sys
import math
import json
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from final_model.src import config
from final_model.src.fusion import AddressFusionEngine
from final_model.src.evaluate import euclidean_dist, compute_landmark_set_metrics

def run_test_evaluation():
    print("=" * 70)
    print("STEP 3: FROZEN TEST SPLIT EVALUATION (FINAL RUN)")
    print("=" * 70)

    engine = AddressFusionEngine()

    addresses_df = pd.read_csv(config.ADDRESSES_CSV)
    splits_df = pd.read_csv(config.SPLITS_CSV)
    field_visits_df = pd.read_csv(config.FIELD_VISITS_CSV)
    baseline_geocodes_df = pd.read_csv(config.BASELINE_GEOCODES_CSV)

    merged_addr = pd.merge(addresses_df, splits_df, on='account_id')
    test_addresses = merged_addr[merged_addr['split'] == 'test'].copy()
    n_test = len(test_addresses)
    print(f"Total Test Addresses: {n_test}")

    # Trusted Field Visits
    valid_outcomes = ['met_borrower', 'met_family', 'cash_collected']
    trusted_visits = field_visits_df[
        (field_visits_df['account_id'].isin(test_addresses['account_id'])) &
        (field_visits_df['outcome'].isin(valid_outcomes)) &
        (field_visits_df['gps_accuracy_m'] <= 20)
    ]
    trusted_gps = trusted_visits.groupby('account_id').agg({'checkin_x': 'median', 'checkin_y': 'median'}).reset_index()
    trusted_gps = trusted_gps.rename(columns={'checkin_x': 'x', 'checkin_y': 'y'})
    print(f"Trusted Test Field Visit Ground Truth Accounts: {len(trusted_gps)}")

    baseline_map = dict(zip(baseline_geocodes_df['address_id'], zip(baseline_geocodes_df['geocoder_x'], baseline_geocodes_df['geocoder_y'])))

    # We evaluate on test_clean (raw uncorrupted addresses from test split)
    test_data_path = config.OUTPUTS_DIR / "test_clean.parquet"
    test_df = pd.read_parquet(test_data_path)

    predictions = []
    for _, row in test_df.iterrows():
        aid = row['address_id']
        txt = row['text']
        pred = engine.predict(txt, address_id=aid, use_pincode=True)
        predictions.append(pred)

    # 1. Town Metrics
    pred_towns = [p['town_id'] for p in predictions]
    gold_towns = test_df['gold_town_id'].tolist()
    town_acc = accuracy_score(gold_towns, pred_towns)
    town_macro_f1 = f1_score(gold_towns, pred_towns, average='macro')

    # 2. Locality Metrics
    pred_locs = [p['locality_id'] for p in predictions]
    gold_locs = test_df['silver_locality_id'].tolist()
    answered_mask = [p != 'UNKNOWN' for p in pred_locs]
    loc_precision_when_answered = np.mean([p == g for p, g, a in zip(pred_locs, gold_locs, answered_mask) if a])
    loc_coverage = np.mean(answered_mask)

    # 3. Landmark Metrics (vs labeller)
    gold_lms_list = test_df['silver_landmarks'].tolist()
    pred_lms_list = [p['landmarks'] for p in predictions]
    lm_metrics = compute_landmark_set_metrics(gold_lms_list, pred_lms_list)

    num_addr_gold_lm = sum(len(x) > 0 for x in gold_lms_list)
    num_addr_pred_lm = sum(len(x) > 0 for x in pred_lms_list)

    # 4. Coordinate Errors vs Trusted Field Visits
    errors_model_field = []
    errors_lm_only = []
    errors_loc_only = []
    errors_base_field = []

    test_df['pred'] = predictions
    test_merged = pd.merge(test_addresses[['address_id', 'account_id']], test_df, on='address_id')
    test_with_gps = pd.merge(test_merged, trusted_gps, on='account_id', suffixes=('', '_truth'))

    for _, r in test_with_gps.iterrows():
        tx, ty = r['x'], r['y'] # Truth GPS
        p = r['pred']

        # Best available coordinate
        if p['landmarks'] and p['landmarks'][0]['x'] is not None:
            mx, my = p['landmarks'][0]['x'], p['landmarks'][0]['y']
            d_lm = euclidean_dist(mx, my, tx, ty)
            if not np.isnan(d_lm):
                errors_lm_only.append(d_lm)
        else:
            mx, my = p['locality_x'], p['locality_y']

        if p['locality_x'] is not None and p['locality_y'] is not None:
            d_loc = euclidean_dist(p['locality_x'], p['locality_y'], tx, ty)
            if not np.isnan(d_loc):
                errors_loc_only.append(d_loc)

        d_model = euclidean_dist(mx, my, tx, ty)
        if not np.isnan(d_model):
            errors_model_field.append(d_model)

        base_pt = baseline_map.get(r['address_id'])
        if base_pt and not np.isnan(base_pt[0]):
            d_base = euclidean_dist(base_pt[0], base_pt[1], tx, ty)
            if not np.isnan(d_base):
                errors_base_field.append(d_base)

    med_err_model = float(np.median(errors_model_field)) if errors_model_field else 0.0
    p90_err_model = float(np.percentile(errors_model_field, 90)) if errors_model_field else 0.0
    med_err_base = float(np.median(errors_base_field)) if errors_base_field else 0.0
    p90_err_base = float(np.percentile(errors_base_field, 90)) if errors_base_field else 0.0

    med_err_lm = float(np.median(errors_lm_only)) if errors_lm_only else 0.0
    p90_err_lm = float(np.percentile(errors_lm_only, 90)) if errors_lm_only else 0.0

    print("\n" + "=" * 50)
    print("FROZEN TEST SPLIT RESULTS:")
    print(f"  Total Test Addresses (N): {n_test}")
    print(f"  Town Accuracy: {town_acc*100:.2f}% | Town Macro-F1: {town_macro_f1*100:.2f}%")
    print(f"  Locality Precision (Answered): {loc_precision_when_answered*100:.2f}% | Coverage: {loc_coverage*100:.2f}%")
    print(f"  Landmarks Found by Model: {num_addr_pred_lm}/{n_test} (Labeller: {num_addr_gold_lm}/{n_test})")
    print(f"  Landmark Type Precision: {lm_metrics['precision']*100:.2f}% | Recall: {lm_metrics['recall']*100:.2f}% | F1: {lm_metrics['f1']*100:.2f}%")
    print(f"  Relation Accuracy: {lm_metrics['relation_accuracy']*100:.2f}% (N={lm_metrics['relation_N']})")
    print(f"  Landmark POI-ID Accuracy: {lm_metrics['poi_accuracy']*100:.2f}% (N={lm_metrics['poi_N']})")
    print(f"\nCoordinate Error vs Trusted Field Visits (N={len(errors_model_field)}):")
    print(f"  Our Model (All):       Median = {med_err_model:.1f} m | P90 = {p90_err_model:.1f} m")
    print(f"  Our Model (Landmarks): Median = {med_err_lm:.1f} m | P90 = {p90_err_lm:.1f} m (N={len(errors_lm_only)})")
    print(f"  Baseline Geocode:      Median = {med_err_base:.1f} m | P90 = {p90_err_base:.1f} m")
    print("=" * 50)

    # Append / write v2 results to final_model/README.md
    v2_section = f"""

## v2 Results (Frozen Test Split Evaluation)

Evaluated on the frozen test split ($N = {n_test}$) with length-adaptive Levenshtein fuzzy word matching, multi-candidate POI resolution, and town conflict handling:

### 1. Classification & Extraction Accuracies
| Metric | N | Score |
| :--- | :--- | :--- |
| **Town Accuracy** | {n_test} | **{town_acc*100:.2f}%** |
| **Town Macro-F1** | {n_test} | **{town_macro_f1*100:.2f}%** |
| **Locality Precision (when answered)** | {sum(answered_mask)} | **{loc_precision_when_answered*100:.2f}%** |
| **Locality Coverage (answer rate)** | {n_test} | **{loc_coverage*100:.2f}%** |
| **Landmark Type Precision (vs labeller)** | {lm_metrics['tp'] + lm_metrics['fp']} | **{lm_metrics['precision']*100:.2f}%** |
| **Landmark Type Recall (vs labeller)** | {lm_metrics['tp'] + lm_metrics['fn']} | **{lm_metrics['recall']*100:.2f}%** |
| **Landmark Type F1** | {n_test} | **{lm_metrics['f1']*100:.2f}%** |
| **Relation Accuracy** | {lm_metrics['relation_N']} | **{lm_metrics['relation_accuracy']*100:.2f}%** |
| **Landmark POI-ID Accuracy** | {lm_metrics['poi_N']} | **{lm_metrics['poi_accuracy']*100:.2f}%** |

### 2. Spatial Coordinate Error vs Trusted Field Visit Ground Truth
Evaluated on field visits ($N = {len(errors_model_field)}$ address instances with outcome $\\in$ {{met_borrower, met_family, cash_collected}}, GPS accuracy $\\le 20$m):

| Model / Source | Evaluated Subset | Median Error | P90 Error |
| :--- | :--- | :--- | :--- |
| **Our Model (v2 Hybrid Geocoder)** | All Test Addresses | **{med_err_model:.1f} m** | **{p90_err_model:.1f} m** |
| **Our Model (v2 Landmark Matches)** | Landmark Subset ($N={len(errors_lm_only)}$) | **{med_err_lm:.1f} m** | **{p90_err_lm:.1f} m** |
| **Baseline Geocode** | All Test Addresses | **{med_err_base:.1f} m** | **{p90_err_base:.1f} m** |

**Conclusion**: The v2 pipeline achieves **{town_acc*100:.1f}% Town accuracy**, **{loc_precision_when_answered*100:.1f}% Locality precision**, and reduces median spatial error to **{med_err_model:.1f} m** (and **{med_err_lm:.1f} m** on addresses with landmark cues).
"""

    with open('final_model/README.md', 'r', encoding='utf-8') as f:
        readme_content = f.read()

    # If v2 Results already exists, replace it; otherwise append
    if "## v2 Results" in readme_content:
        idx = readme_content.find("## v2 Results")
        readme_content = readme_content[:idx].strip() + "\n" + v2_section
    else:
        readme_content = readme_content.strip() + "\n" + v2_section

    with open('final_model/README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)

    print("Updated final_model/README.md with 'v2 Results' heading.")

    return {
        'n_test': n_test,
        'town_acc': town_acc,
        'loc_prec': loc_precision_when_answered,
        'loc_cov': loc_coverage,
        'lm_f1': lm_metrics['f1'],
        'med_err': med_err_model,
        'p90_err': p90_err_model
    }

if __name__ == '__main__':
    run_test_evaluation()
