import pandas as pd

splits = pd.read_csv("final_model/data_ref/splits.csv")
addrs = pd.read_csv("final_model/data_ref/addresses.csv")
hand = pd.read_csv("final_model/outputs/handcheck_30.csv")

val = addrs.merge(splits[splits["split"] == "validation"], on="account_id")
val = val[~val["address_id"].isin(hand["address_id"])]
fresh = val.sample(n=30, random_state=42)[["address_id", "raw_address" if "raw_address" in val.columns else "address_text"]]
fresh.columns = ["address_id", "raw_address"]
fresh.to_csv("final_model/outputs/handcheck_fresh_30.csv", index=False)
print(fresh.to_string())
print("Saved", len(fresh), "rows")