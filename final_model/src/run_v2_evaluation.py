"""
Evaluation and Comparison Script for Final Model v2.
Runs on:
1. 5 sample addresses from sample_output.json
2. 20 validation addresses from data_ref/splits.csv
3. 10 hand-made fuzzy spellings (testing before vs after)
Generates final_model/outputs/sample_output_v2.json and metrics for change_report.md.
"""
import io
import sys
import json
import pandas as pd

# UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from final_model.src.predict import predict
from final_model.src.labeller import DeterministicLabeller, preprocess_address

def run_evaluation():
    labeller = DeterministicLabeller()

    # -------------------------------------------------------------
    # 1. 5 Sample Addresses
    # -------------------------------------------------------------
    with open('final_model/outputs/sample_output.json', encoding='utf-8') as f:
        sample_5 = json.load(f)

    sample_5_results = []
    print("\n=======================================================")
    print("1. EVALUATION ON 5 SAMPLE ADDRESSES")
    print("=======================================================")
    for item in sample_5:
        aid = item['address_id']
        raw = item['raw_address']
        res = predict(raw, address_id=aid)
        lbl = labeller.label_address(raw)

        print(f"Address ID: {aid}")
        print(f"  Raw: {raw}")
        print(f"  Model  -> Town: {res['town_name']} | Locality: {res['locality_name']} | Landmarks: {len(res['landmarks'])}")
        print(f"  Label  -> Town: {lbl['town_id']} | Locality: {lbl['locality_name']} | Landmarks: {len(lbl['landmarks'])}")
        sample_5_results.append(res)

    # -------------------------------------------------------------
    # 2. 20 Validation Addresses
    # -------------------------------------------------------------
    splits = pd.read_csv('final_model/data_ref/splits.csv')
    addrs = pd.read_csv('final_model/data_ref/addresses.csv')
    val_addrs = addrs.merge(splits[splits['split'] == 'validation'], on='account_id')
    
    val_20 = val_addrs.head(20)
    val_20_results = []
    
    town_matches = 0
    loc_matches = 0
    lm_matches = 0
    
    print("\n=======================================================")
    print("2. EVALUATION ON 20 VALIDATION ADDRESSES")
    print("=======================================================")
    for i, (_, row) in enumerate(val_20.iterrows()):
        aid = row['address_id']
        raw = row['address_text']
        res = predict(raw, address_id=aid)
        lbl = labeller.label_address(raw)

        # Match checks
        lbl_town_name = labeller.town_id_to_name.get(lbl['town_id'], lbl['town_id'])
        t_ok = (res['town_name'] == lbl_town_name)
        l_ok = (res['locality_name'] == lbl['locality_name'])
        
        # Compare landmark types
        res_lm_types = set(lm['landmark_type'] for lm in res['landmarks'])
        lbl_lm_types = set(lm['landmark_type'] for lm in lbl['landmarks'])
        lm_ok = (res_lm_types == lbl_lm_types)

        if t_ok: town_matches += 1
        if l_ok: loc_matches += 1
        if lm_ok: lm_matches += 1

        print(f"[{i+1}/20] {aid}")
        print(f"  Raw: {raw[:60]}...")
        print(f"  Town: Model={res['town_name']} vs Labeller={lbl_town_name} -> {t_ok}")
        print(f"  Locality: Model={res['locality_name']} vs Labeller={lbl['locality_name']} -> {l_ok}")
        print(f"  Landmarks: Model={sorted(list(res_lm_types))} vs Labeller={sorted(list(lbl_lm_types))} -> {lm_ok}")
        val_20_results.append(res)

    print(f"\n20 Val Correctness vs Labeller:")
    print(f"  Town Correctness: {town_matches}/20 ({town_matches/20*100:.1f}%)")
    print(f"  Locality Correctness: {loc_matches}/20 ({loc_matches/20*100:.1f}%)")
    print(f"  Landmark Type Correctness: {lm_matches}/20 ({lm_matches/20*100:.1f}%)")

    # -------------------------------------------------------------
    # 3. 10 Hand-Made Fuzzy Spellings (Before vs After)
    # -------------------------------------------------------------
    # We will simulate "Before" (exact regex / simple labeller without fuzzy)
    # and compare with "After" (predict() with fuzzy n-gram + abbreviation expansion)
    fuzzy_tests = [
        ("F01", "silver oak lyt 4th cross 8th main nr govt school 980301", "Silver Oak Layout", "govt_school", "Navanagara East"),
        ("F02", "devggarh ngr gali 4 near church patel nagar 970202", "Patel Nagar", "church", "Devgarh Nagar"),
        ("F03", "10th cross gandhi nagr hanuman mandr kaveripura 960102", "Gandhi Nagar", "hanuman_temple", "Kaveripura"),
        ("F04", "kuvempu layt 3rd cross opp bus stand kaveripura 960102", "Kuvempu Layout", "bus_stop", "Kaveripura"),
        ("F05", "house 12 indira nagr nr ration shop devgarh ngr 970201", "Indira Nagar", "ration_shop", "Devgarh Nagar"),
        ("F06", "7th mn rd 2nd x ashoka layt kaveripura 960104", "Ashoka Layout", None, "Kaveripura"),
        ("F07", "bhd medicals basava nagr kaveripura 960101", "Basava Nagar", "medical_store", "Kaveripura"),
        ("F08", "5th cross vinayaka layt nr pani ki tanki 960104", "Vinayaka Nagar", "water_tank", "Kaveripura"),
        ("F09", "opp doodh dairy nehru colny devgarh ngr 970203", "Nehru Colony", "milk_dairy", "Devgarh Nagar"),
        ("F10", "nr kalyana mantapa shanthi nagr navanagara 980302", "Shanthi Nagar", "community_hall", "Navanagara East")
    ]

    print("\n=======================================================")
    print("3. EVALUATION ON 10 HAND-MADE FUZZY SPELLINGS")
    print("=======================================================")
    fuzzy_results = []
    before_after_log = []

    for fid, raw, expected_loc, expected_lm, expected_town in fuzzy_tests:
        # Before: baseline raw regex search without fuzzy distance
        # Check what the raw labeller would get
        lbl = labeller.label_address(raw)
        lbl_town = labeller.town_id_to_name.get(lbl['town_id'], lbl['town_id'])
        lbl_loc = lbl['locality_name']
        lbl_lms = [lm['landmark_type'] for lm in lbl['landmarks']]

        # After: new predict engine
        res = predict(raw, address_id=fid)
        res_town = res['town_name']
        res_loc = res['locality_name']
        res_lms = [lm['landmark_type'] for lm in res['landmarks']]

        print(f"\n[{fid}] Input: \"{raw}\"")
        print(f"  Target Expected: Town={expected_town}, Loc={expected_loc}, LM={expected_lm}")
        print(f"  BEFORE (Baseline) -> Town={lbl_town}, Loc={lbl_loc}, LMs={lbl_lms}")
        print(f"  AFTER  (Enhanced) -> Town={res_town}, Loc={res_loc}, LMs={res_lms}")
        print(f"  Locality Resolved: {res_loc} | Coords: ({res['locality_x']}, {res['locality_y']})")
        print(f"  Landmarks Resolved: {[{'name': lm['landmark_name'], 'type': lm['landmark_type'], 'mode': lm['resolution_mode'], 'cnt': lm['candidate_count']} for lm in res['landmarks']]}")

        before_after_log.append({
            'id': fid,
            'input': raw,
            'target': {'town': expected_town, 'locality': expected_loc, 'landmark': expected_lm},
            'before': {'town': lbl_town, 'locality': lbl_loc, 'landmarks': lbl_lms},
            'after': {'town': res_town, 'locality': res_loc, 'landmarks': res_lms}
        })
        fuzzy_results.append(res)

    # -------------------------------------------------------------
    # 4. Save sample_output_v2.json
    # -------------------------------------------------------------
    # Combine the 5 sample addresses, 20 validation addresses, and 10 fuzzy test addresses
    final_output = {
        'sample_5_addresses': sample_5_results,
        'validation_20_addresses': val_20_results,
        'fuzzy_10_test_addresses': fuzzy_results
    }

    with open('final_model/outputs/sample_output_v2.json', 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)

    print("\nSaved comprehensive outputs to final_model/outputs/sample_output_v2.json")

    return {
        'sample_5': sample_5_results,
        'val_20': val_20_results,
        'fuzzy_10': fuzzy_results,
        'val_metrics': {
            'town_acc': town_matches / 20.0,
            'loc_acc': loc_matches / 20.0,
            'lm_acc': lm_matches / 20.0
        },
        'before_after': before_after_log
    }

if __name__ == '__main__':
    run_evaluation()
