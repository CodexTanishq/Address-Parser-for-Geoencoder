import pandas as pd
from final_model.src.fusion import AddressFusionEngine
from final_model.src import config

eng = AddressFusionEngine()

OUT = config.OUTPUTS_DIR
MASTER = OUT.parent / "data_ref" / "master_data_3_locations_v5.csv"

master = pd.read_csv(MASTER, encoding="utf-8-sig")
gold = pd.read_csv(OUT / "handcheck_fresh_30_gold.csv", encoding="utf-8-sig")
addr = pd.read_csv(OUT / "model_output_new.csv", encoding="utf-8-sig")[["address_id", "raw_address"]]
data = gold.merge(addr, on="address_id")


def expected_pois(town, locality, lm_name):
    """Returns (mode, set of POI names, count) from the master file."""
    if lm_name == "UNKNOWN" or town == "OUT":
        return "none", set(), 0
    t = master[(master["town_name"] == town) & (master["landmark_name"] == lm_name)]
    in_loc = t[t["nearest_locality_name"] == locality]
    if locality != "UNKNOWN" and len(in_loc) > 0:
        return "in_locality", set(in_loc["landmark_name"]), len(in_loc)
    return "candidates", set(t["landmark_name"]), len(t)


t_ok = l_ok = lm_ok = 0
for _, row in data.iterrows():
    res = eng.predict(row["raw_address"])
    town_ok = res["town_name"] == row["expected_town_name"]
    loc_ok = res["locality_name"] == row["expected_locality_name"]

    exp_mode, exp_names, exp_n = expected_pois(
        row["expected_town_name"], row["expected_locality_name"], row["expected_landmark_name"]
    )
    lms = res["landmarks"]
    got_names = {lm["landmark_name"] for lm in lms}
    got_n = max([lm["candidate_count"] for lm in lms], default=0)
    got_modes = {lm["resolution_mode"] for lm in lms}

    if exp_mode == "none":
        lm_ok_row = len(lms) == 0
    else:
        lm_ok_row = (got_names == exp_names) and (got_n == exp_n) and (got_modes == {exp_mode})

    t_ok += town_ok
    l_ok += loc_ok
    lm_ok += lm_ok_row

    if not (town_ok and loc_ok and lm_ok_row):
        print("\nWRONG:", row["address_id"], row["raw_address"])
        print("  model:", res["town_name"], "|", res["locality_name"], "|", got_names, got_n, got_modes)
        print("  gold :", row["expected_town_name"], "|", row["expected_locality_name"], "|", exp_names, exp_n, exp_mode)

n = len(data)
print(f"\nN = {n}")
print(f"Town     : {t_ok}/{n} = {100*t_ok/n:.1f}%")
print(f"Locality : {l_ok}/{n} = {100*l_ok/n:.1f}%")
print(f"Landmark (strict: name + count + mode) : {lm_ok}/{n} = {100*lm_ok/n:.1f}%")