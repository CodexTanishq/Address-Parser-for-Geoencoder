from pathlib import Path
import pandas as pd
from src import config
from src.remark_parser import FieldVisitRemarkParser, parse_field_visits


def test_remark_parser_features():
    print("=" * 70)
    print("  TESTING FIELD VISIT REMARK PARSER (src/remark_parser.py)")
    print("=" * 70)

    parser = FieldVisitRemarkParser()

    # 1. Test Landmark Extraction
    sample_remark_1 = "yaaru illa, beega haakide; nija mane Milk Dairy hinde ide, 2 cross munde"
    res1 = parser.parse_record("locked_premises", sample_remark_1)
    print("\n[Test Case 1: Complex Landmark + Directional + Correction]")
    print(f"Remark: {sample_remark_1}")
    print(f"Extracted Landmarks:       {res1['extracted_landmarks']}")
    print(f"Directional Vectors:       {res1['directional_vectors']}")
    print(f"Address Correction Found:  {res1['has_address_correction']}")
    print(f"Corrected Clause:          {res1['corrected_address_clause']}")
    print(f"Contradiction Detected:    {res1['is_contradiction']}")

    assert len(res1["extracted_landmarks"]) > 0, "Failed to extract landmark from Test Case 1"
    assert res1["has_address_correction"] is True, "Failed to flag address correction in Test Case 1"

    # 2. Test Secondary Multi-Landmarks
    sample_remark_2 = "customer dukaan pe mila; actual house behind Mosque, near Water Tank, 2 lanes ahead"
    res2 = parser.parse_record("met_borrower", sample_remark_2)
    print("\n[Test Case 2: Multi-Landmark + Spatial Offset]")
    print(f"Remark: {sample_remark_2}")
    print(f"Extracted Landmarks:       {res2['extracted_landmarks']}")
    print(f"Directional Vectors:       {res2['directional_vectors']}")
    assert len(res2["extracted_landmarks"]) >= 2, "Expected multiple landmarks in Test Case 2"

    # 3. Test Outcome vs Remark Contradiction Detection
    sample_remark_3 = "ghar band tha, yaaru illa, beega haakide"
    res3 = parser.parse_record("met_borrower", sample_remark_3)
    print("\n[Test Case 3: Integrity Contradiction Flag]")
    print(f"Outcome: met_borrower | Remark: {sample_remark_3}")
    print(f"Contradiction Detected:    {res3['is_contradiction']}")
    print(f"Contradiction Reason:      {res3['contradiction_reason']}")
    assert res3["is_contradiction"] is True, "Failed to catch obvious contradiction in Test Case 3"

    sample_remark_4 = "customer se mila, cash collect kiya"
    res4 = parser.parse_record("address_not_traceable", sample_remark_4)
    print("\n[Test Case 4: Reverse Contradiction Flag]")
    print(f"Outcome: address_not_traceable | Remark: {sample_remark_4}")
    print(f"Contradiction Detected:    {res4['is_contradiction']}")
    print(f"Contradiction Reason:      {res4['contradiction_reason']}")
    assert res4["is_contradiction"] is True, "Failed to catch reverse contradiction in Test Case 4"

    # 4. Test on real field_visits.csv batch
    if config.FIELD_VISITS_CSV.exists():
        print("\n" + "-" * 70)
        print("  PROCESSING REAL BATCH FROM data/field_visits.csv")
        print("-" * 70)
        df_parsed = parse_field_visits(limit=50)
        print(f"Successfully processed {len(df_parsed)} field visits.")
        
        contradictions = df_parsed[df_parsed["is_contradiction"] == True]
        corrections = df_parsed[df_parsed["has_address_correction"] == True]
        with_landmarks = df_parsed[df_parsed["extracted_landmarks"] != ""]

        print(f"Field visits with extracted landmarks:   {len(with_landmarks)}")
        print(f"Field visits with address corrections:   {len(corrections)}")
        print(f"Field visits with outcome contradictions: {len(contradictions)}")

        output_path = config.OUTPUTS_DIR / "field_visits_parsed_sample.csv"
        df_parsed.to_csv(output_path, index=False)
        print(f"Sample output saved to {output_path}")

    print("\nAll Remark Parser tests passed successfully!")


if __name__ == "__main__":
    test_remark_parser_features()
