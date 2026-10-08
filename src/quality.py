import json
from typing import List, Dict, Any
import pandas as pd
from pathlib import Path
from src import config
from src.parser import ParsedAddress


def compute_pipeline_statistics(
    parsed_addresses: List[ParsedAddress],
    weak_labels_df: pd.DataFrame,
) -> Dict[str, Any]:
    """Compute detailed quality and coverage statistics across the parsed addresses."""
    total_addresses = len(parsed_addresses)
    valid_pincodes = sum(1 for p in parsed_addresses if p.pincode is not None)
    resolved_localities = sum(1 for p in parsed_addresses if not p.is_locality_unresolved)
    unresolved_localities = sum(1 for p in parsed_addresses if p.is_locality_unresolved)
    ambiguous_localities = sum(1 for p in parsed_addresses if p.is_ambiguous)

    house_number_count = sum(1 for p in parsed_addresses if p.house_number is not None)
    street_count = sum(1 for p in parsed_addresses if len(p.streets) > 0)
    ward_count = sum(1 for p in parsed_addresses if len(p.wards) > 0)
    landmark_count = sum(1 for p in parsed_addresses if len(p.landmarks) > 0)

    # Token label statistics
    token_conf_counts = weak_labels_df["confidence"].value_counts().to_dict() if not weak_labels_df.empty else {}
    unlabeled_tokens = (weak_labels_df["label"] == "O").sum() if not weak_labels_df.empty else 0
    total_tokens = len(weak_labels_df) if not weak_labels_df.empty else 0

    entity_token_counts = (
        weak_labels_df[weak_labels_df["entity"] != "O"]["entity"].value_counts().to_dict()
        if not weak_labels_df.empty
        else {}
    )

    stats = {
        "dataset_summary": {
            "total_addresses_processed": total_addresses,
            "valid_pincodes": valid_pincodes,
            "valid_pincodes_pct": round(100.0 * valid_pincodes / total_addresses, 2) if total_addresses else 0,
            "resolved_localities": resolved_localities,
            "resolved_localities_pct": round(100.0 * resolved_localities / total_addresses, 2) if total_addresses else 0,
            "unresolved_localities": unresolved_localities,
            "ambiguous_localities": ambiguous_localities,
        },
        "entity_coverage": {
            "house_number_coverage": house_number_count,
            "house_number_pct": round(100.0 * house_number_count / total_addresses, 2) if total_addresses else 0,
            "street_coverage": street_count,
            "street_pct": round(100.0 * street_count / total_addresses, 2) if total_addresses else 0,
            "ward_coverage": ward_count,
            "ward_pct": round(100.0 * ward_count / total_addresses, 2) if total_addresses else 0,
            "landmark_coverage": landmark_count,
            "landmark_pct": round(100.0 * landmark_count / total_addresses, 2) if total_addresses else 0,
        },
        "weak_supervision_confidence": {
            "high_confidence_tokens": int(token_conf_counts.get(config.CONF_HIGH, 0)),
            "medium_confidence_tokens": int(token_conf_counts.get(config.CONF_MEDIUM, 0)),
            "ambiguous_tokens": int(token_conf_counts.get(config.CONF_AMBIGUOUS, 0)),
            "low_confidence_tokens": int(token_conf_counts.get(config.CONF_LOW, 0)),
            "unlabeled_tokens_O": int(unlabeled_tokens),
            "total_tokens": int(total_tokens),
        },
        "entity_token_counts": entity_token_counts,
    }

    return stats


def generate_human_inspection_sample(
    parsed_addresses: List[ParsedAddress],
    target_count: int = 120,
) -> pd.DataFrame:
    """
    Produce a curated inspection sample of at least 100 representative examples
    including:
    - successful examples
    - failure/unresolved examples
    - ambiguous examples
    - multilingual (Kannada, Devanagari) examples
    - misspelled locality examples
    """
    selected: Dict[str, ParsedAddress] = {}

    def add_item(p: ParsedAddress):
        if p.address_id not in selected and len(selected) < target_count:
            selected[p.address_id] = p

    # 1. Indic scripts (Kannada, Devanagari)
    for p in parsed_addresses:
        if any(s in ["Kannada", "Devanagari"] for s in p.scripts):
            add_item(p)
            if len([x for x in selected.values() if "Kannada" in x.scripts]) >= 20:
                break

    # 2. Ambiguous examples
    for p in parsed_addresses:
        if p.is_ambiguous:
            add_item(p)

    # 3. Unresolved locality examples
    for p in parsed_addresses:
        if p.is_locality_unresolved:
            add_item(p)
            if len([x for x in selected.values() if x.is_locality_unresolved]) >= 30:
                break

    # 4. Misspelled/fuzzy resolved examples
    for p in parsed_addresses:
        if p.locality_metadata.get("match_type") == "fuzzy_gazetteer":
            add_item(p)
            if len([x for x in selected.values() if x.locality_metadata.get("match_type") == "fuzzy_gazetteer"]) >= 25:
                break

    # 5. High-confidence fully resolved examples with landmarks and house numbers
    for p in parsed_addresses:
        if p.house_number and p.streets and p.landmarks and p.locality_name:
            add_item(p)
            if len(selected) >= target_count:
                break

    # Fill remainder to reach target_count
    for p in parsed_addresses:
        if len(selected) >= target_count:
            break
        add_item(p)

    records = []
    for p in selected.values():
        records.append({
            "address_id": p.address_id,
            "town_id": p.town_id,
            "original_address": p.address_text,
            "transliterated_address": p.address_transliterated,
            "scripts": "+".join(p.scripts),
            "house_number": p.house_number or "",
            "streets": " | ".join(p.streets),
            "ward": " | ".join(p.wards),
            "landmark": " | ".join(p.landmarks),
            "locality": p.locality_name or "",
            "town": p.town_name or "",
            "pincode": p.pincode or "",
            "confidence": p.overall_confidence,
            "locality_match_type": p.locality_metadata.get("match_type", ""),
            "other_details": p.other_details,
        })

    return pd.DataFrame(records)
