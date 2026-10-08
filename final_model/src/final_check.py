import pandas as pd
from final_model.src.fusion import AddressFusionEngine
from final_model.src import config

eng = AddressFusionEngine()

# Part 1: fuzzy spellings
fuzzy = [
    "4th cross, 8th main, silver oak lyt - 980301",
    "gali 4 devggarh ngr near church patel nagar - 970202",
    "10th cross gandhi nagr hanuman mandr kaveripura - 960102",
    "kuvempu layt 3rd cross opp bus stop kaveripura - 960102",
]

print("=== FUZZY SPELLINGS ===")
for text in fuzzy:
    res = eng.predict(text)
    print("\nINPUT:", text)
    print("TOWN:", res["town_name"], "| LOCALITY:", res["locality_name"])
    for lm in res["landmarks"]:
        print("  ", lm["landmark_name"], "|", lm["resolution_mode"],
              "| count:", lm["candidate_count"])
    if not res["landmarks"]:
        print("   no landmarks")

# Part 2: hand-check 30
print("\n=== HAND-CHECK 30 ===")
check = pd.read_csv(config.OUTPUTS_DIR / "handcheck_30.csv")
gold = pd.read_csv(config.OUTPUTS_DIR / "handcheck_30_gold.csv")
gold = gold.rename(columns={
    "expected_town_name": "town_name_gold",
    "expected_locality_name": "locality_name_gold",
    "expected_landmark_name": "landmark_name_gold",
})

print("check columns:", list(check.columns))
print("gold columns:", list(gold.columns))

# Use the address text from handcheck_30.csv, matched by row order
# after confirming the address_id values line up.
merged = check.merge(gold, on="address_id")
print("merged columns:", list(merged.columns))

t_ok = l_ok = lm_ok = 0
for _, row in merged.iterrows():
    addr = row["raw_address_x"] if "raw_address_x" in merged.columns else row["raw_address"]
    res = eng.predict(addr)
    town_ok = res["town_name"] == row["town_name_gold"]
    loc_ok = res["locality_name"] == row["locality_name_gold"]
    lm_names = [lm["landmark_name"] for lm in res["landmarks"]]

    if row["landmark_name_gold"] == "UNKNOWN":
        lm_ok_row = len(lm_names) == 0
    else:
        lm_ok_row = row["landmark_name_gold"] in lm_names

    t_ok += town_ok
    l_ok += loc_ok
    lm_ok += lm_ok_row

    if not (town_ok and loc_ok and lm_ok_row):
        print("\nWRONG:", row["address_id"], addr)
        print("  model:", res["town_name"], "|", res["locality_name"], "|", lm_names)
        print("  gold :", row["town_name_gold"], "|", row["locality_name_gold"], "|", row["landmark_name_gold"])

n = len(merged)
print(f"\nN = {n}")
print(f"Town     : {t_ok}/{n} = {100*t_ok/n:.1f}%")
print(f"Locality : {l_ok}/{n} = {100*l_ok/n:.1f}%")
print(f"Landmark : {lm_ok}/{n} = {100*lm_ok/n:.1f}%")