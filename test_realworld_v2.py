import pandas as pd
from pathlib import Path
from src.inference import AddressNERPredictor, format_address_to_english

# Define model checkpoint paths
MODEL_V1_PATH = Path("models/muril_address_ner_v1_backup")
MODEL_V2_PATH = Path("models/muril_address_ner_v2_multilandmark")
INPUT_CSV = "realworld_test_50_addresses.csv"
OUTPUT_CSV = "outputs/realworld_parsed_v2_comparison.csv"

def run_realworld_comparison():
    print("=" * 80)
    print("  RUNNING REAL-WORLD EVALUATION: MODEL V1 vs MODEL V2 (MULTI-LANDMARK)")
    print("=" * 80)

    # 1. Initialize predictors for both models
    print(f"[+] Loading Model v1 from: {MODEL_V1_PATH}")
    predictor_v1 = AddressNERPredictor(model_dir=MODEL_V1_PATH)

    print(f"[+] Loading Model v2 from: {MODEL_V2_PATH}")
    predictor_v2 = AddressNERPredictor(model_dir=MODEL_V2_PATH)

    # 2. Read the 50 unseen real-world test addresses
    df = pd.read_csv(INPUT_CSV)
    print(f"[+] Loaded {len(df)} test addresses from {INPUT_CSV}\n")

    results = []

    for idx, row in df.iterrows():
        aid = row["address_id"]
        raw_text = row["raw_address"]
        lang = row.get("language_mix", "UNKNOWN")

        # Model v1 Prediction
        res_v1 = predictor_v1.predict(raw_text)
        struct_v1 = res_v1.get("structured_address", {})
        landmarks_v1 = struct_v1.get("landmarks", [])
        if isinstance(landmarks_v1, str):
            landmarks_v1 = [landmarks_v1] if landmarks_v1 else []

        # Model v2 Prediction
        res_v2 = predictor_v2.predict(raw_text)
        struct_v2 = res_v2.get("structured_address", {})
        landmarks_v2 = struct_v2.get("landmarks", [])
        if isinstance(landmarks_v2, str):
            landmarks_v2 = [landmarks_v2] if landmarks_v2 else []

        # Convert Model v2 output to Natural English Navigational Sentence
        english_desc = format_address_to_english(struct_v2)

        results.append({
            "address_id": aid,
            "language_mix": lang,
            "raw_address": raw_text,
            "house_number": struct_v2.get("house_number") or "",
            "streets": ", ".join(struct_v2.get("streets", [])) if isinstance(struct_v2.get("streets"), list) else (struct_v2.get("streets") or ""),
            "locality": struct_v2.get("locality") or "",
            "town": struct_v2.get("town") or "",
            "pincode": struct_v2.get("pincode") or "",
            "v1_landmarks_count": len(landmarks_v1),
            "v1_landmarks": " | ".join(landmarks_v1),
            "v2_landmarks_count": len(landmarks_v2),
            "v2_landmarks": " | ".join(landmarks_v2),
            "english_navigational_statement": english_desc
        })

    # 3. Save comparison output to CSV
    out_df = pd.DataFrame(results)
    Path("outputs").mkdir(exist_ok=True)
    try:
        out_df.to_csv(OUTPUT_CSV, index=False)
    except PermissionError:
        fallback_path = "outputs/realworld_parsed_v2_comparison_clean.csv"
        out_df.to_csv(fallback_path, index=False)
        print(f"[!] Warning: '{OUTPUT_CSV}' is locked by another application. Saved to '{fallback_path}'.")

    # 4. Print Summary Statistics
    print("-" * 80)
    print("  EVALUATION SUMMARY & COMPARISON METRICS")
    print("-" * 80)
    print(f"Total Test Addresses        : {len(out_df)}")
    print(f"Model v1 Total Landmarks    : {out_df['v1_landmarks_count'].sum()}")
    print(f"Model v2 Total Landmarks    : {out_df['v2_landmarks_count'].sum()}  <-- (+{out_df['v2_landmarks_count'].sum() - out_df['v1_landmarks_count'].sum()} extra landmarks captured)")
    print(f"Addresses with >1 Landmark  : {(out_df['v2_landmarks_count'] > 1).sum()}")
    print(f"\nDetailed CSV comparison saved to: {OUTPUT_CSV}\n")

    # Display sample side-by-side output for inspection
    print("=" * 80)
    print("  SAMPLE MULTI-LANDMARK PARSING & ENGLISH TRANSFORMATION")
    print("=" * 80)
    sample_df = out_df[out_df["v2_landmarks_count"] > 0].head(5)
    for _, r in sample_df.iterrows():
        print(f"[{r['address_id']}] RAW: {r['raw_address']}")
        print(f"   Model v1 Landmarks : {r['v1_landmarks']}")
        print(f"   Model v2 Landmarks : {r['v2_landmarks']}")
        print(f"   English Statement  : {r['english_navigational_statement']}")
        print("-" * 80)

if __name__ == "__main__":
    run_realworld_comparison()