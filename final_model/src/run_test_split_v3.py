"""
Frozen test split run (v3 landmark rule). Run ONCE. Do not tune after.
Does NOT touch README.md or old result files.
Output: outputs/test_v3_results.json

Changes vs run_test_split_eval.py:
 - Landmark metrics count one entry per detected TYPE (not one per candidate POI).
 - POI-id accuracy: gold poi_id must be inside the returned POI set for that type.
 - Coordinate error reported 3 ways for landmark addresses:
     first      = first returned POI (what the old script did, arbitrary)
     centroid   = mean x,y of the returned POIs of the first detected type
     (and split by mode: in_locality vs candidates)
   Addresses with no landmark use the locality centroid.
"""
import io
import sys
import json
import math
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from final_model.src import config
from final_model.src.fusion import AddressFusionEngine


def dist(x1, y1, x2, y2):
    vals = [x1, y1, x2, y2]
    if any(v is None for v in vals) or any(np.isnan(v) for v in vals):
        return np.nan
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def by_type(lms):
    """Group landmark entries by type, keeping order of first appearance."""
    groups = {}
    for lm in lms:
        groups.setdefault(lm['landmark_type'], []).append(lm)
    return groups


def landmark_metrics(gold_list, pred_list):
    tp = fp = fn = 0
    rel_ok = rel_n = poi_ok = poi_n = 0
    for gold_lms, pred_lms in zip(gold_list, pred_list):
        g_groups = by_type(gold_lms)
        p_groups = by_type(pred_lms)
        for t in p_groups:
            if t in g_groups:
                tp += 1
            else:
                fp += 1
        for t in g_groups:
            if t not in p_groups:
                fn += 1
        for g in gold_lms:
            t = g['landmark_type']
            if t in p_groups:
                p_first = p_groups[t][0]
                rel_n += 1
                if g.get('relation') == p_first.get('relation'):
                    rel_ok += 1
                if g.get('poi_id') is not None:
                    poi_n += 1
                    ids = {str(p.get('poi_id')) for p in p_groups[t]}
                    if str(g.get('poi_id')) in ids:
                        poi_ok += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        'tp': tp, 'fp': fp, 'fn': fn,
        'precision': round(prec, 4), 'recall': round(rec, 4), 'f1': round(f1, 4),
        'relation_accuracy': round(rel_ok / rel_n, 4) if rel_n else 0.0, 'relation_N': rel_n,
        'poi_in_set_accuracy': round(poi_ok / poi_n, 4) if poi_n else 0.0, 'poi_N': poi_n,
    }


def summarize(errs):
    if not errs:
        return {'N': 0, 'median_m': None, 'p90_m': None}
    return {'N': len(errs),
            'median_m': round(float(np.median(errs)), 1),
            'p90_m': round(float(np.percentile(errs, 90)), 1)}


def run():
    print("=" * 70)
    print("FROZEN TEST SPLIT - V3 LANDMARK RULE (FINAL RUN)")
    print("=" * 70)

    engine = AddressFusionEngine()

    addresses_df = pd.read_csv(config.ADDRESSES_CSV)
    splits_df = pd.read_csv(config.SPLITS_CSV)
    visits_df = pd.read_csv(config.FIELD_VISITS_CSV)
    base_df = pd.read_csv(config.BASELINE_GEOCODES_CSV)

    merged = pd.merge(addresses_df, splits_df, on='account_id')
    test_addr = merged[merged['split'] == 'test'].copy()
    n_test = len(test_addr)
    print(f"Total Test Addresses: {n_test}")

    ok_out = ['met_borrower', 'met_family', 'cash_collected']
    trusted = visits_df[
        (visits_df['account_id'].isin(test_addr['account_id'])) &
        (visits_df['outcome'].isin(ok_out)) &
        (visits_df['gps_accuracy_m'] <= 20)
    ]
    gps = trusted.groupby('account_id').agg({'checkin_x': 'median', 'checkin_y': 'median'}).reset_index()
    gps = gps.rename(columns={'checkin_x': 'x', 'checkin_y': 'y'})
    print(f"Trusted GPS accounts: {len(gps)}")

    base_map = dict(zip(base_df['address_id'], zip(base_df['geocoder_x'], base_df['geocoder_y'])))

    test_df = pd.read_parquet(config.OUTPUTS_DIR / "test_clean.parquet")
    preds = [engine.predict(r['text'], address_id=r['address_id'], use_pincode=True)
             for _, r in test_df.iterrows()]

    # Town
    pred_towns = [p['town_id'] for p in preds]
    gold_towns = test_df['gold_town_id'].tolist()
    town_acc = accuracy_score(gold_towns, pred_towns)
    town_f1 = f1_score(gold_towns, pred_towns, average='macro')

    # Locality
    pred_locs = [p['locality_id'] for p in preds]
    gold_locs = test_df['silver_locality_id'].tolist()
    answered = [p != 'UNKNOWN' for p in pred_locs]
    loc_prec = float(np.mean([p == g for p, g, a in zip(pred_locs, gold_locs, answered) if a]))
    loc_cov = float(np.mean(answered))

    # Landmarks
    gold_lms = test_df['silver_landmarks'].tolist()
    pred_lms = [p['landmarks'] for p in preds]
    lm = landmark_metrics(gold_lms, pred_lms)
    n_gold_lm = sum(len(x) > 0 for x in gold_lms)
    n_pred_lm = sum(len(x) > 0 for x in pred_lms)

    # Mode counts (per address, first detected type)
    mode_counts = {'in_locality': 0, 'candidates': 0, 'none': 0}
    for p in preds:
        if p['landmarks']:
            mode_counts[p['landmarks'][0]['resolution_mode']] += 1
        else:
            mode_counts['none'] += 1

    # Coordinates
    test_df['pred'] = preds
    tm = pd.merge(test_addr[['address_id', 'account_id']], test_df, on='address_id')
    tw = pd.merge(tm, gps, on='account_id', suffixes=('', '_truth'))

    e_first, e_centroid, e_base = [], [], []
    e_in_loc, e_cand, e_locality_only, e_nolm = [], [], [], []
    for _, r in tw.iterrows():
        tx, ty = r['x'], r['y']
        p = r['pred']

        d = dist(p['locality_x'], p['locality_y'], tx, ty)
        if not np.isnan(d):
            e_locality_only.append(d)

        if p['landmarks']:
            first_type = p['landmarks'][0]['landmark_type']
            pois = [l for l in p['landmarks'] if l['landmark_type'] == first_type]
            fx, fy = pois[0]['poi_x'], pois[0]['poi_y']
            cx = float(np.mean([l['poi_x'] for l in pois]))
            cy = float(np.mean([l['poi_y'] for l in pois]))
            d_first = dist(fx, fy, tx, ty)
            d_cent = dist(cx, cy, tx, ty)
            if not np.isnan(d_first):
                e_first.append(d_first)
            if not np.isnan(d_cent):
                e_centroid.append(d_cent)
                (e_in_loc if pois[0]['resolution_mode'] == 'in_locality' else e_cand).append(d_cent)
        else:
            d_fallback = dist(p['locality_x'], p['locality_y'], tx, ty)
            if not np.isnan(d_fallback):
                e_first.append(d_fallback)
                e_centroid.append(d_fallback)
                e_nolm.append(d_fallback)

        bp = base_map.get(r['address_id'])
        if bp and not np.isnan(bp[0]):
            d_b = dist(bp[0], bp[1], tx, ty)
            if not np.isnan(d_b):
                e_base.append(d_b)

    results = {
        'N_test': n_test,
        'town_accuracy': round(town_acc, 4),
        'town_macro_f1': round(town_f1, 4),
        'locality_precision_answered': round(loc_prec, 4),
        'locality_coverage': round(loc_cov, 4),
        'landmark_addresses_labeller': n_gold_lm,
        'landmark_addresses_model': n_pred_lm,
        'landmark_metrics': lm,
        'resolution_mode_counts': mode_counts,
        'coord_error_m': {
            'N_gps_points': len(tw),
            'model_centroid_all': summarize(e_centroid),
            'model_first_poi_all': summarize(e_first),
            'model_centroid_in_locality': summarize(e_in_loc),
            'model_centroid_candidates': summarize(e_cand),
            'model_no_landmark_locality_fallback': summarize(e_nolm),
            'locality_centroid_only_all': summarize(e_locality_only),
            'baseline_geocode': summarize(e_base),
        },
    }

    print("\n" + "=" * 50)
    print(f"Town Acc: {town_acc*100:.2f}% | Macro-F1: {town_f1*100:.2f}%")
    print(f"Locality Precision (answered): {loc_prec*100:.2f}% | Coverage: {loc_cov*100:.2f}%")
    print(f"Landmarks found: {n_pred_lm}/{n_test} (labeller: {n_gold_lm}/{n_test})")
    print(f"Landmark type P/R/F1: {lm['precision']*100:.2f} / {lm['recall']*100:.2f} / {lm['f1']*100:.2f}")
    print(f"Relation acc: {lm['relation_accuracy']*100:.2f}% (N={lm['relation_N']})")
    print(f"POI-in-set acc: {lm['poi_in_set_accuracy']*100:.2f}% (N={lm['poi_N']})")
    print(f"Modes (first type per address): {mode_counts}")
    print("\nCoordinate error vs trusted GPS (metres):")
    for k, v in results['coord_error_m'].items():
        if isinstance(v, dict):
            print(f"  {k:38s} N={v['N']:4d} | median={v['median_m']} | p90={v['p90_m']}")
    print("=" * 50)

    out = config.OUTPUTS_DIR / "test_v3_results.json"
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    print("Saved:", out)


if __name__ == '__main__':
    run()