import pandas as pd
from final_model.src.fusion import AddressFusionEngine
from final_model.src import config

# Change these two names to run on another file.
# The input CSV needs the columns: address_id, raw_address
INPUT_NAME = "model_output_new.csv"
OUTPUT_NAME = "landmark_rows_fresh30.csv"

eng = AddressFusionEngine()
src = pd.read_csv(config.OUTPUTS_DIR / INPUT_NAME, encoding="utf-8-sig")

rows = []
for _, r in src.iterrows():
    res = eng.predict(r["raw_address"], address_id=r["address_id"])
    base = {
        "address_id": r["address_id"],
        "raw_address": r["raw_address"],
        "town_id": res["town_id"],
        "town_name": res["town_name"],
        "locality_id": res["locality_id"],
        "locality_name": res["locality_name"],
        "locality_x": res["locality_x"],
        "locality_y": res["locality_y"],
    }
    if res["landmarks"]:
        for lm in res["landmarks"]:
            row = dict(base)
            row.update({
                "landmark_name": lm["landmark_name"],
                "landmark_type": lm["landmark_type"],
                "poi_id": lm["poi_id"],
                "poi_x": lm["poi_x"],
                "poi_y": lm["poi_y"],
                "resolution_mode": lm["resolution_mode"],
                "candidate_count": lm["candidate_count"],
            })
            rows.append(row)
    else:
        row = dict(base)
        row.update({
            "landmark_name": "",
            "landmark_type": "",
            "poi_id": "",
            "poi_x": "",
            "poi_y": "",
            "resolution_mode": "none",
            "candidate_count": 0,
        })
        rows.append(row)

out = pd.DataFrame(rows)
out.to_csv(config.OUTPUTS_DIR / OUTPUT_NAME, index=False, encoding="utf-8-sig")
print("Addresses:", len(src))
print("Rows written:", len(out))
print(out["resolution_mode"].value_counts().to_string())
print("Saved:", config.OUTPUTS_DIR / OUTPUT_NAME)