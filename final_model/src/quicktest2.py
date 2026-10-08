from final_model.src.fusion import AddressFusionEngine

eng = AddressFusionEngine()

tests = [
    "Devgarh Nagar, Ambedkar Nagar, near ganesh temple - 970202",
    "Devgarh Nagar, near ganesh temple - 970202",
    "Devgarh Nagar, Gandhi Basti, near park - 970202",
    "Devgarh Nagar, Indira Colony, near bus stop - 970202",
]

for text in tests:
    res = eng.predict(text)
    print("\nINPUT:", text)
    print("TOWN:", res["town_name"], "| LOCALITY:", res["locality_name"])
    for lm in res["landmarks"]:
        print("  ", lm["landmark_name"], "|", lm["resolution_mode"],
              "| count:", lm["candidate_count"],
              "| x:", lm["poi_x"], "y:", lm["poi_y"])
    if not res["landmarks"]:
        print("   no landmarks returned")