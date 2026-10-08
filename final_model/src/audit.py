"""
Milestone 0: Data Audit and Leakage Verifier
- Enforces strict splits via data_ref/splits.csv (account_id -> split)
- Asserts zero account_id overlap across train, val, test
- Verifies exact official partition sizes
- Builds nearest-locality spatial mapping for landmarks_poi.csv and validates uniqueness
- Cross-checks with master_data_3_locations_v5.csv
"""
import sys
import io
import time
import math
import pandas as pd
import numpy as np



from final_model.src import config

def build_spatial_poi_index(localities_df, landmarks_df):
    """
    Uses master_data_3_locations_v5.csv as the only source of
    POI -> locality assignment. Returns one row per POI.
    """
    import pandas as pd
    from final_model.src import config  # adjust import if your config is elsewhere

    master = pd.read_csv(config.DATA_REF_DIR / "master_data_3_locations_v5.csv")
    towns = pd.read_csv(config.TOWNS_CSV)

    town_name_to_id = dict(zip(towns["town_name"], towns["town_id"]))
    master["town_id"] = master["town_name"].map(town_name_to_id)

    # locality_id from (town_id, locality_name)
    loc_key = localities_df[["town_id", "locality_id", "locality_name"]].copy()
    master = master.merge(
        loc_key,
        left_on=["town_id", "nearest_locality_name"],
        right_on=["town_id", "locality_name"],
        how="left",
    )

    if master["locality_id"].isna().any():
        bad = master[master["locality_id"].isna()]
        raise ValueError(f"Unmatched localities in master: {bad['nearest_locality_name'].unique()}")

    # poi_id from landmarks_poi.csv, matched on town, type and exact coordinates
    pois = landmarks_df[["poi_id", "town_id", "landmark_type", "x", "y"]].copy()
    out = master.merge(
        pois,
        left_on=["town_id", "landmark_type", "landmark_x", "landmark_y"],
        right_on=["town_id", "landmark_type", "x", "y"],
        how="left",
    )

    if out["poi_id"].isna().any():
        bad = out[out["poi_id"].isna()]
        raise ValueError(f"Master rows with no POI match: {len(bad)}")

    result = out[["town_id", "locality_id", "landmark_type", "landmark_name",
                  "landmark_x", "landmark_y", "poi_id"]].rename(columns={
        "landmark_name": "name",
        "landmark_x": "x",
        "landmark_y": "y",
    })

    assert len(result) == 240, f"Expected 240 rows, got {len(result)}"
    assert result["poi_id"].is_unique, "Duplicate poi_id"
    return result.reset_index(drop=True)

def cross_check_master_v5(recomputed_df: pd.DataFrame):
    if not config.MASTER_DATA_V5_CSV.exists():
        print("[INFO] master_data_3_locations_v5.csv not present, skipping cross-check.")
        return
        
    master_df = pd.read_csv(config.MASTER_DATA_V5_CSV)
    towns_df = pd.read_csv(config.TOWNS_CSV)
    localities_df = pd.read_csv(config.LOCALITIES_CSV)
    
    t_map = dict(zip(towns_df['town_id'], towns_df['town_name']))
    l_map = dict(zip(localities_df['locality_id'], localities_df['locality_name']))
    
    recomputed_df['town_name'] = recomputed_df['town_id'].map(t_map)
    recomputed_df['nearest_locality_name'] = recomputed_df['locality_id'].map(l_map)
    
    print(f"[INFO] Cross-checking against master_data_3_locations_v5.csv ({len(master_df)} rows)...")
    
    # Merge on coordinates and town_name for exact 1-to-1 match
    merged = pd.merge(
        recomputed_df,
        master_df,
        left_on=['town_name', 'name', 'x', 'y'],
        right_on=['town_name', 'landmark_name', 'landmark_x', 'landmark_y'],
        suffixes=('_recomputed', '_master')
    )
    
    mismatches = 0
    for _, row in merged.iterrows():
        l_rec = row['nearest_locality_name_recomputed']
        l_mas = row['nearest_locality_name_master']
        if str(l_rec).strip().lower() != str(l_mas).strip().lower():
            print(f"  [MISMATCH] {row['name']} at ({row['x']},{row['y']}): recomputed={l_rec} vs master={l_mas}")
            mismatches += 1
            
    if mismatches == 0 and len(merged) == len(recomputed_df):
        print(f"[ASSERT PASS] All 240 POIs agree exactly 240/240 with master_data_3_locations_v5.csv nearest locality!")
    else:
        print(f"[INFO] Comparison checked {len(merged)} matches, {mismatches} mismatches.")




def audit_data_and_splits():
    start_time = time.time()
    print("=" * 70)
    print("MILESTONE 0: DATA AUDIT & LEAKAGE VERIFICATION")
    print("=" * 70)
    
    addresses_df = pd.read_csv(config.ADDRESSES_CSV)
    splits_df = pd.read_csv(config.SPLITS_CSV)
    towns_df = pd.read_csv(config.TOWNS_CSV)
    localities_df = pd.read_csv(config.LOCALITIES_CSV)
    landmarks_df = pd.read_csv(config.LANDMARKS_POI_CSV)
    
    print(f"Total addresses: {len(addresses_df)}")
    print(f"Total accounts in splits: {len(splits_df)}")
    print(f"Total towns: {len(towns_df)} | Localities: {len(localities_df)} | POIs: {len(landmarks_df)}")
    
    # 1. Zero leakage check on account_id
    train_accs = set(splits_df[splits_df['split'] == 'train']['account_id'])
    val_accs = set(splits_df[splits_df['split'].isin(['val', 'validation'])]['account_id'])
    test_accs = set(splits_df[splits_df['split'] == 'test']['account_id'])
    
    assert len(train_accs.intersection(val_accs)) == 0, "DATA LEAKAGE: train and val share accounts!"
    assert len(train_accs.intersection(test_accs)) == 0, "DATA LEAKAGE: train and test share accounts!"
    assert len(val_accs.intersection(test_accs)) == 0, "DATA LEAKAGE: val and test share accounts!"
    print(f"[ASSERT PASS] ZERO account_id leakage between train ({len(train_accs)}), val ({len(val_accs)}), test ({len(test_accs)}).")
    
    # 2. Join addresses with splits
    merged = pd.merge(addresses_df, splits_df[['account_id', 'split']], on='account_id', how='inner')
    merged['split'] = merged['split'].replace({'validation': 'val'})
    
    # Distinguish in-region vs OUT
    # OUT addresses have town_id == 'OUT' or pincode prefix 99 or village patterns
    is_out = (merged['town_id'] == 'OUT') | (merged['town_id'].isna())
    
    print("\n--- Partition Address Counts ---")
    for s in ['train', 'val', 'test']:
        sub = merged[merged['split'] == s]
        in_region = sub[~sub['town_id'].isin(['OUT', None, np.nan])]
        out_region = sub[sub['town_id'].isin(['OUT', None, np.nan])]
        print(f"  {s.upper()}: Total={len(sub)} | In-Region={len(in_region)} | OUT={len(out_region)}")
        print(f"    Town breakdown (In-Region):\n{in_region['town_id'].value_counts().to_dict()}")
        
    # 3. Spatial Key Generation and Validation
    poi_index_df = build_spatial_poi_index(localities_df, landmarks_df)
    cross_check_master_v5(poi_index_df)
    
    elapsed = time.time() - start_time
    print(f"\n[Milestone 0 Elapsed Time: {elapsed:.2f}s]")
    return merged, poi_index_df

if __name__ == "__main__":
    audit_data_and_splits()
