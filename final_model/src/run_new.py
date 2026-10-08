import pandas as pd
from final_model.src.fusion import AddressFusionEngine
from final_model.src import config

eng = AddressFusionEngine()
df = pd.read_csv(config.OUTPUTS_DIR / "handcheck_fresh_30.csv")

rows = []
for _, r in df.iterrows():
    res = eng.predict(r["raw_address"], address_id=r["address_id"])
    lms = res["landmarks"]
    rows.append({
        "address_id": r["address_id"],
        "raw_address": r["raw_address"],
        "town_name": res["town_name"],
        "locality_name": res["locality_name"],
        "landmark_names": "; ".join(sorted(set(lm["landmark_name"] for lm in lms))),
        "landmark_types": "; ".join(sorted(set(lm["landmark_type"] for lm in lms))),
        "n_landmark_candidates": len(lms),
    })

out = pd.DataFrame(rows)
out.to_csv(config.OUTPUTS_DIR / "model_output_new.csv", index=False)
print(out.to_string())