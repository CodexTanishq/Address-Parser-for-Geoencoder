import sys
import io
import json
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output encoding on Windows consoles
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src import config
from src.utils import set_seed, load_raw_data
from src.parser import AddressParser, ParsedAddress
from src.weak_labels import build_weak_datasets
from src.quality import compute_pipeline_statistics, generate_human_inspection_sample


def run_pipeline() -> None:
    print("=" * 70)
    print("  WEAK-SUPERVISION INDIAN ADDRESS NER PIPELINE (HIGH-PRECISION TEACHER)")
    print("=" * 70)

    # 1. Reproducibility
    set_seed(config.RANDOM_SEED)

    # 2. Ensure directories exist
    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 3. Load Immutable Raw Data
    print("\n[1/6] Loading immutable raw data...")
    addresses_df, towns_df, localities_df = load_raw_data()
    print(f"  Loaded {len(addresses_df)} addresses, {len(towns_df)} towns, {len(localities_df)} localities.")

    # Target subset: T1, T2, T3 (2880 addresses)
    target_addresses = addresses_df[addresses_df["town_id"].isin(["T1", "T2", "T3"])].copy()
    print(f"  Target subset (T1/T2/T3): {len(target_addresses)} addresses")

    # 4. Initialize Parser
    print("\n[2/6] Initializing modular high-precision deterministic parser...")
    parser = AddressParser(towns_df, localities_df)

    # 5. Parse Addresses
    print("\n[3/6] Parsing addresses and resolving entities...")
    parsed_results = []
    for _, row in target_addresses.iterrows():
        res = parser.parse(
            address_id=str(row["address_id"]),
            address_text=str(row["address_text"]),
            town_id=str(row["town_id"]),
        )
        parsed_results.append(res)
    print(f"  Successfully parsed {len(parsed_results)} addresses.")

    # 6. Build Weak Labels and BIO Dataset
    print("\n[4/6] Generating weak supervision labels and BIO sequences...")
    weak_spans_df, weak_labels_df, bio_dataset_df, unresolved_df, ambiguous_df = (
        build_weak_datasets(parsed_results)
    )

    # Build parsed_addresses DataFrame
    parsed_records = []
    for p in parsed_results:
        parsed_records.append({
            "address_id": p.address_id,
            "town_id": p.town_id,
            "address_text": p.address_text,
            "address_transliterated": p.address_transliterated,
            "address_normalized": p.address_normalized,
            "scripts": "+".join(p.scripts),
            "pincode": p.pincode or "",
            "town_name": p.town_name or "",
            "locality_name": p.locality_name or "",
            "house_number": p.house_number or "",
            "streets": " | ".join(p.streets),
            "wards": " | ".join(p.wards),
            "landmarks": " | ".join(p.landmarks),
            "other_details": p.other_details,
            "overall_confidence": p.overall_confidence,
            "is_ambiguous": p.is_ambiguous,
            "is_locality_unresolved": p.is_locality_unresolved,
        })
    parsed_df = pd.DataFrame(parsed_records)

    # 7. Compute Quality Metrics & Generate Human Inspection Sample
    print("\n[5/6] Computing statistics and generating inspection sample...")
    stats = compute_pipeline_statistics(parsed_results, weak_labels_df)
    inspection_sample_df = generate_human_inspection_sample(parsed_results, target_count=120)

    # 8. Export Artifacts
    print("\n[6/6] Writing output datasets to disk...")
    parsed_df.to_csv(config.PARSED_ADDRESSES_CSV, index=False)
    weak_spans_df.to_csv(config.WEAK_SPANS_CSV, index=False)
    weak_labels_df.to_csv(config.WEAK_LABELS_CSV, index=False)
    bio_dataset_df.to_csv(config.BIO_DATASET_CSV, index=False)
    unresolved_df.to_csv(config.UNRESOLVED_ADDRESSES_CSV, index=False)
    ambiguous_df.to_csv(config.AMBIGUOUS_ADDRESSES_CSV, index=False)
    inspection_sample_df.to_csv(config.HUMAN_INSPECTION_CSV, index=False)

    with open(config.PARSER_STATISTICS_JSON, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print(f"  -> {config.PARSED_ADDRESSES_CSV} ({len(parsed_df)} rows)")
    print(f"  -> {config.WEAK_SPANS_CSV} ({len(weak_spans_df)} rows)")
    print(f"  -> {config.WEAK_LABELS_CSV} ({len(weak_labels_df)} tokens)")
    print(f"  -> {config.BIO_DATASET_CSV} ({len(bio_dataset_df)} rows)")
    print(f"  -> {config.UNRESOLVED_ADDRESSES_CSV} ({len(unresolved_df)} rows)")
    print(f"  -> {config.AMBIGUOUS_ADDRESSES_CSV} ({len(ambiguous_df)} rows)")
    print(f"  -> {config.HUMAN_INSPECTION_CSV} ({len(inspection_sample_df)} sample rows)")
    print(f"  -> {config.PARSER_STATISTICS_JSON}")

    # Display Report
    print("\n" + "=" * 70)
    print("  PARSER QUALITY AND COVERAGE REPORT")
    print("=" * 70)
    ds = stats["summary"] if "summary" in stats else stats["dataset_summary"]
    cov = stats["entity_coverage"]
    conf = stats["weak_supervision_confidence"]

    print(f"Total T1/T2/T3 Addresses: {ds['total_addresses_processed']}")
    print(f"Valid Pincodes:           {ds['valid_pincodes']} ({ds['valid_pincodes_pct']}%)")
    print(f"Resolved Localities:      {ds['resolved_localities']} ({ds['resolved_localities_pct']}%)")
    print(f"Unresolved Localities:    {ds['unresolved_localities']}")
    print(f"Ambiguous Localities:     {ds['ambiguous_localities']}")
    print("-" * 50)
    print(f"House Numbers Extracted:  {cov['house_number_coverage']} ({cov['house_number_pct']}%)")
    print(f"Streets Extracted:        {cov['street_coverage']} ({cov['street_pct']}%)")
    print(f"Wards Extracted:          {cov['ward_coverage']} ({cov['ward_pct']}%)")
    print(f"Landmarks Extracted:      {cov['landmark_coverage']} ({cov['landmark_pct']}%)")
    print("-" * 50)
    print(f"High-Confidence Tokens:   {conf['high_confidence_tokens']}")
    print(f"Unlabeled Tokens (O):     {conf['unlabeled_tokens_O']}")
    print(f"Total Labeled Spans:      {len(weak_spans_df)}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline()
