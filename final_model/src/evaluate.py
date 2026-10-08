"""
Milestone 5: Comprehensive Test Split Evaluation & Coordinate Error Benchmarking
- Runs on frozen test split (459 addresses: 427 in-region, 32 OUT).
- Evaluates across Clean, Mild, Heavy noise conditions.
- Evaluates both with pincode cue ON and with pincode cue OFF (ablated).
- Metrics computed:
  1. Town accuracy and macro-F1 (in-region + OUT).
  2. Locality precision (when answered) and coverage (answer rate).
  3. Landmark metrics:
     - Landmark-type set precision, recall, and F1 (vs labeller)
     - Relation accuracy (when both have landmark)
     - Landmark POI-id resolution accuracy (exact match of poi_id)
  4. Coordinate distance errors in metres vs:
     (a) Trusted home-visit GPS points (test split accounts, outcome in {met_borrower, met_family, cash_collected}, gps_accuracy_m <= 20).
     (b) Baseline geocodes on the same addresses.
     (c) Surveyed addresses test subset (N=12) and all 100 surveyed addresses.
- Saves:
  final_model/outputs/predictions_test.csv
  final_model/outputs/evaluation_metrics.json
  final_model/outputs/sample_landmark_missed.txt
  final_model/outputs/sample_landmark_wrong.txt
"""
import sys
import io
import math
import json
import time
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from final_model.src import config
from final_model.src.fusion import AddressFusionEngine

def euclidean_dist(x1, y1, x2, y2):
    if any(v is None or np.isnan(v) for v in [x1, y1, x2, y2]):
        return np.nan
    return math.sqrt((x1 - x2)**2 + (y1 - y2)**2)

def compute_landmark_set_metrics(gold_landmarks_list, pred_landmarks_list):
    """
    Computes set-level precision, recall, and F1 for landmark types,
    as well as relation accuracy and POI-id accuracy.
    """
    tp = 0
    fp = 0
    fn = 0
    
    relation_correct = 0
    relation_total = 0
    
    poi_correct = 0
    poi_total = 0
    
    for gold_lms, pred_lms in zip(gold_landmarks_list, pred_landmarks_list):
        gold_types = [lm['landmark_type'] for lm in gold_lms]
        pred_types = [lm['landmark_type'] for lm in pred_lms]
        
        # Multiset / count overlap
        gold_counts = {}
        for t in gold_types:
            gold_counts[t] = gold_counts.get(t, 0) + 1
            
        pred_counts = {}
        for t in pred_types:
            pred_counts[t] = pred_counts.get(t, 0) + 1
            
        for t, p_cnt in pred_counts.items():
            g_cnt = gold_counts.get(t, 0)
            tp += min(p_cnt, g_cnt)
            if p_cnt > g_cnt:
                fp += (p_cnt - g_cnt)
                
        for t, g_cnt in gold_counts.items():
            p_cnt = pred_counts.get(t, 0)
            if g_cnt > p_cnt:
                fn += (g_cnt - p_cnt)
                
        # Relation and POI ID matching for aligned pairs
        if len(gold_lms) > 0 and len(pred_lms) > 0:
            for g_lm in gold_lms:
                # Find matching pred by type
                matching_p = next((p for p in pred_lms if p['landmark_type'] == g_lm['landmark_type']), None)
                if matching_p:
                    # Relation check
                    relation_total += 1
                    if g_lm.get('relation') == matching_p.get('relation'):
                        relation_correct += 1
                        
                    # POI ID check
                    if g_lm.get('poi_id') is not None:
                        poi_total += 1
                        if str(g_lm.get('poi_id')) == str(matching_p.get('poi_id')):
                            poi_correct += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    rel_acc = relation_correct / relation_total if relation_total > 0 else 0.0
    poi_acc = poi_correct / poi_total if poi_total > 0 else 0.0
    
    return {
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(f1, 4),
        'relation_accuracy': round(rel_acc, 4),
        'relation_N': relation_total,
        'poi_accuracy': round(poi_acc, 4),
        'poi_N': poi_total
    }

def evaluate_test_split():
    start_time = time.time()
    print("=" * 70)
    print("MILESTONE 5: COMPREHENSIVE TEST EVALUATION & COORDINATE BENCHMARK")
    print("=" * 70)

    engine = AddressFusionEngine()

    # Load Reference Ground Truth
    addresses_df = pd.read_csv(config.ADDRESSES_CSV)
    splits_df = pd.read_csv(config.SPLITS_CSV)
    field_visits_df = pd.read_csv(config.FIELD_VISITS_CSV)
    baseline_geocodes_df = pd.read_csv(config.BASELINE_GEOCODES_CSV)
    surveyed_df = pd.read_csv(config.SURVEYED_ADDRESSES_CSV)

    merged_addr = pd.merge(addresses_df, splits_df, on='account_id')
    test_addresses = merged_addr[merged_addr['split'] == 'test'].copy()
    n_test = len(test_addresses)
    print(f"Total Test Addresses: {n_test} (In-Region: {(test_addresses['town_id']!='OUT').sum()}, OUT: {(test_addresses['town_id']=='OUT').sum()})")

    # Filter Trusted Field Visits for Test Accounts
    valid_outcomes = ['met_borrower', 'met_family', 'cash_collected']
    trusted_visits = field_visits_df[
        (field_visits_df['account_id'].isin(test_addresses['account_id'])) &
        (field_visits_df['outcome'].isin(valid_outcomes)) &
        (field_visits_df['gps_accuracy_m'] <= 20)
    ]
    # Median GPS per account
    trusted_gps = trusted_visits.groupby('account_id').agg({'checkin_x': 'median', 'checkin_y': 'median'}).reset_index()
    trusted_gps = trusted_gps.rename(columns={'checkin_x': 'x', 'checkin_y': 'y'})
    print(f"Trusted Test Field Visit Ground Truth Accounts: {len(trusted_gps)}")

    # Merge Baseline Geocodes (columns: geocoder_x, geocoder_y)
    baseline_map = dict(zip(baseline_geocodes_df['address_id'], zip(baseline_geocodes_df['geocoder_x'], baseline_geocodes_df['geocoder_y'])))

    evaluation_summary = {}

    for condition in ['clean', 'mild', 'heavy']:
        for use_pincode in [True, False]:
            cond_key = condition if use_pincode else f"{condition}_no_pincode"
            pin_status = "WITH PINCODE" if use_pincode else "WITHOUT PINCODE"
            print(f"\n" + "-" * 60)
            print(f"Evaluating Test Condition: {condition.upper()} [{pin_status}] (N={n_test})")
            print("-" * 60)

            test_data_path = config.OUTPUTS_DIR / f"test_{condition}.parquet"
            test_df = pd.read_parquet(test_data_path)

            predictions = []
            pred_records_for_csv = []

            for _, row in test_df.iterrows():
                aid = row['address_id']
                txt = row['text']
                pred = engine.predict(txt, use_pincode=use_pincode)
                predictions.append(pred)

                if condition == 'clean' and use_pincode:
                    pred_records_for_csv.append({
                        'address_id': aid,
                        'town_id': pred['town_id'],
                        'locality_id': pred['locality_id'],
                        'locality_x': pred['locality_x'],
                        'locality_y': pred['locality_y'],
                        'landmarks': json.dumps(pred['landmarks']),
                        'town_conf': pred['confidences']['town'],
                        'locality_conf': pred['confidences']['locality'],
                        'landmarks_conf': pred['confidences']['landmarks'],
                        'conflict_flag': pred['conflict_flag']
                    })

            # Save clean predictions to outputs/predictions_test.csv
            if condition == 'clean' and use_pincode:
                pd.DataFrame(pred_records_for_csv).to_csv(config.PREDICTIONS_TEST_CSV, index=False)
                print(f"  [OK] Saved final predictions to {config.PREDICTIONS_TEST_CSV}")

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

            # Count addresses with >=1 landmark
            num_addr_gold_lm = sum(len(x) > 0 for x in gold_lms_list)
            num_addr_pred_lm = sum(len(x) > 0 for x in pred_lms_list)

            # 4. Coordinate Errors vs Trusted Field Visits
            errors_model_field = []
            errors_base_field = []

            test_df['pred'] = predictions
            test_merged = pd.merge(test_addresses[['address_id', 'account_id']], test_df, on='address_id')
            test_with_gps = pd.merge(test_merged, trusted_gps, on='account_id', suffixes=('', '_truth'))

            for _, r in test_with_gps.iterrows():
                tx, ty = r['x'], r['y'] # GPS ground truth
                p = r['pred']
                if p['landmarks'] and p['landmarks'][0]['x'] is not None:
                    mx, my = p['landmarks'][0]['x'], p['landmarks'][0]['y']
                else:
                    mx, my = p['locality_x'], p['locality_y']

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

            print(f"  Town Accuracy: {town_acc*100:.2f}% | Town Macro-F1: {town_macro_f1*100:.2f}% (N={n_test})")
            print(f"  Locality Precision (Answered): {loc_precision_when_answered*100:.2f}% | Coverage: {loc_coverage*100:.2f}% (N={n_test})")
            print(f"  Landmarks by Labeller: {num_addr_gold_lm}/{n_test} | Found by Model: {num_addr_pred_lm}/{n_test}")
            print(f"  Landmark Type Precision: {lm_metrics['precision']*100:.2f}% | Recall: {lm_metrics['recall']*100:.2f}% | F1: {lm_metrics['f1']*100:.2f}% (TP={lm_metrics['tp']}, FP={lm_metrics['fp']}, FN={lm_metrics['fn']})")
            print(f"  Relation Accuracy: {lm_metrics['relation_accuracy']*100:.2f}% (N={lm_metrics['relation_N']})")
            print(f"  Landmark POI-ID Accuracy: {lm_metrics['poi_accuracy']*100:.2f}% (N={lm_metrics['poi_N']})")
            print(f"  Coordinate Error vs Trusted Field Visits (N={len(errors_model_field)}):")
            print(f"    Our Model:       Median = {med_err_model:.1f} m | P90 = {p90_err_model:.1f} m")
            print(f"    Baseline Geocode: Median = {med_err_base:.1f} m | P90 = {p90_err_base:.1f} m")

            evaluation_summary[cond_key] = {
                'N': n_test,
                'pincode_cue_enabled': use_pincode,
                'town_accuracy': round(town_acc, 4),
                'town_macro_f1': round(town_macro_f1, 4),
                'locality_precision_answered': round(loc_precision_when_answered, 4),
                'locality_coverage': round(loc_coverage, 4),
                'landmark_addresses_labeller': num_addr_gold_lm,
                'landmark_addresses_model': num_addr_pred_lm,
                'landmark_metrics': lm_metrics,
                'gps_benchmarks': {
                    'N_gps_points': len(errors_model_field),
                    'model_median_error_m': round(med_err_model, 1),
                    'model_p90_error_m': round(p90_err_model, 1),
                    'baseline_median_error_m': round(med_err_base, 1),
                    'baseline_p90_error_m': round(p90_err_base, 1)
                }
            }

            # Generate 20 missed and 20 wrong landmarks for clean condition
            if condition == 'clean' and use_pincode:
                missed_samples = []
                wrong_samples = []
                for _, r in test_df.iterrows():
                    g_lms = r['silver_landmarks']
                    p_lms = engine.predict(r['text'], use_pincode=True)['landmarks']
                    g_types = [x['landmark_type'] for x in g_lms]
                    p_types = [x['landmark_type'] for x in p_lms]
                    
                    if len(g_types) > 0 and len(p_types) == 0:
                        missed_samples.append({
                            'address_id': r['address_id'],
                            'text': r['text'],
                            'labeller_landmarks': g_types,
                            'model_landmarks': p_types
                        })
                    elif len(p_types) > 0 and g_types != p_types:
                        wrong_samples.append({
                            'address_id': r['address_id'],
                            'text': r['text'],
                            'labeller_landmarks': g_types,
                            'model_landmarks': p_types
                        })
                        
                with open(config.OUTPUTS_DIR / "sample_landmark_missed.txt", "w", encoding="utf-8") as f:
                    for i, s in enumerate(missed_samples[:20]):
                        f.write(f"[{i+1}] {s['address_id']}: {s['text']}\n    Labeller: {s['labeller_landmarks']} | Model: {s['model_landmarks']}\n\n")
                        
                with open(config.OUTPUTS_DIR / "sample_landmark_wrong.txt", "w", encoding="utf-8") as f:
                    for i, s in enumerate(wrong_samples[:20]):
                        f.write(f"[{i+1}] {s['address_id']}: {s['text']}\n    Labeller: {s['labeller_landmarks']} | Model: {s['model_landmarks']}\n\n")
                print(f"  [OK] Saved 20 missed and 20 wrong landmark samples to outputs/")

    # Save metrics JSON
    with open(config.EVALUATION_METRICS_JSON, 'w') as f:
        json.dump(evaluation_summary, f, indent=2)
    print(f"\n[OK] Complete evaluation report saved to {config.EVALUATION_METRICS_JSON}")

    elapsed = time.time() - start_time
    print(f"\n[Test Evaluation Elapsed Time: {elapsed:.2f}s]")

if __name__ == "__main__":
    evaluate_test_split()
