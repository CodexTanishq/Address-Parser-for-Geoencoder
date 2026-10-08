"""
Script to run predict() on 5 addresses from addresses.csv with a landmark and locality,
under light noise. Formats output and saves to final_model/outputs/sample_output.json.
"""
import sys
import io
import json
import pandas as pd
from final_model.src import config
from final_model.src.predict import predict

def generate_sample_output():
    # Load towns map to get town_name
    towns_df = pd.read_csv(config.TOWNS_CSV)
    town_id_to_name = dict(zip(towns_df['town_id'], towns_df['town_name']))
    
    # Load mild noise test split (addresses from addresses.csv with light noise)
    mild_df = pd.read_parquet(config.OUTPUTS_DIR / "test_mild.parquet")
    
    selected_outputs = []
    
    for _, row in mild_df.iterrows():
        aid = row['address_id']
        addr_text = row['text']
        
        pred = predict(addr_text)
        
        # Must have locality with coordinates
        if not pred.get('locality_coordinates'):
            continue
            
        loc_coords = pred['locality_coordinates']
        loc_x = loc_coords.get('x')
        loc_y = loc_coords.get('y')
        if loc_x is None or loc_y is None:
            continue
            
        # Must have landmark(s) with coordinates
        valid_landmarks = []
        for lm in pred.get('landmarks', []):
            px = lm.get('x')
            py = lm.get('y')
            if px is not None and py is not None:
                valid_landmarks.append({
                    "landmark_name": lm.get('poi_name'),
                    "landmark_type": lm.get('landmark_type'),
                    "relation": lm.get('relation'),
                    "poi_x": px,
                    "poi_y": py,
                    "resolution_mode": lm.get('resolution_mode')
                })
                
        if len(valid_landmarks) == 0:
            continue
            
        t_id = pred.get('town_id')
        t_name = town_id_to_name.get(t_id, t_id)
        
        selected_outputs.append({
            "address_id": aid,
            "raw_address": addr_text,
            "town_name": t_name,
            "locality_name": pred.get('locality_name'),
            "locality_x": loc_x,
            "locality_y": loc_y,
            "landmarks": valid_landmarks
        })
        
        if len(selected_outputs) == 5:
            break
            
    out_file = config.OUTPUTS_DIR / "sample_output.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(selected_outputs, f, indent=2, ensure_ascii=False)
        
    print(json.dumps(selected_outputs, indent=2, ensure_ascii=True))
    return selected_outputs


if __name__ == "__main__":
    generate_sample_output()
