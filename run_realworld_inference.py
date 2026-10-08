import pandas as pd
from src.inference import AddressNERPredictor

# Initialize predictor
predictor = AddressNERPredictor()

# Load 50 real-world addresses
df = pd.read_csv("realworld_test_50_addresses.csv")

parsed_results = []

for idx, row in df.iterrows():
    raw_text = row["raw_address"]
    
    # Run prediction
    res = predictor.predict(raw_text)
    
    # Extract from 'structured_address' sub-dictionary
    parsed = res.get("structured_address", {})
    
    parsed_results.append({
        "address_id": row["address_id"],
        "raw_address": raw_text,
        "house_number": parsed.get("house_number") or "",
        "streets": ", ".join(parsed.get("streets", [])) if isinstance(parsed.get("streets"), list) else (parsed.get("streets") or ""),
        "locality": parsed.get("locality") or "",
        "town": parsed.get("town") or "",
        "pincode": parsed.get("pincode") or ""
    })

# Save corrected output CSV
output_df = pd.DataFrame(parsed_results)
output_df.to_csv("realworld_parsed_results_fixed.csv", index=False)
print("Updated results saved to realworld_parsed_results_fixed.csv!")