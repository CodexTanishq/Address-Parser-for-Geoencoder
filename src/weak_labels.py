import json
from typing import List, Tuple, Dict, Any
import pandas as pd
from src import config
from src.parser import ParsedAddress
from src.bio import spans_to_bio, validate_bio_sequence


def build_weak_datasets(
    parsed_addresses: List[ParsedAddress],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build all export datasets from parsed addresses:
    1. weak_spans_df (span-level representations)
    2. weak_labels_df (token-level representations)
    3. bio_dataset_df (sentence/sequence level representations for NER)
    4. unresolved_addresses_df
    5. ambiguous_addresses_df
    """
    span_records = []
    token_records = []
    bio_records = []
    unresolved_records = []
    ambiguous_records = []

    for pa in parsed_addresses:
        # 1. Span records
        for s in pa.spans:
            span_records.append({
                "address_id": pa.address_id,
                "town_id": pa.town_id,
                "entity_type": s.entity_type,
                "entity_text": s.text,
                "start": s.start,
                "end": s.end,
                "confidence": s.confidence,
                "source": s.source,
                "match_score": s.match_score,
            })

        # 2. BIO Tokenization (using HIGH confidence spans for training)
        tokens, bio_labels, token_meta = spans_to_bio(
            pa.address_text, pa.spans, confidence_filter=config.CONF_HIGH
        )
        assert validate_bio_sequence(bio_labels), f"Invalid BIO sequence for {pa.address_id}"

        # Token records
        for tm in token_meta:
            token_records.append({
                "address_id": pa.address_id,
                "token": tm["token"],
                "label": tm["label"],
                "entity": tm["entity"],
                "confidence": tm["confidence"],
                "source": tm["source"],
                "match_score": tm["match_score"],
            })

        # BIO Sequence record
        bio_records.append({
            "address_id": pa.address_id,
            "town_id": pa.town_id,
            "address_text": pa.address_text,
            "address_transliterated": pa.address_transliterated,
            "address_normalized": pa.address_normalized,
            "tokens": json.dumps(tokens, ensure_ascii=False),
            "bio_labels": json.dumps(bio_labels, ensure_ascii=False),
            "overall_confidence": pa.overall_confidence,
            "is_ambiguous": pa.is_ambiguous,
            "is_locality_unresolved": pa.is_locality_unresolved,
        })

        # 3. Track Unresolved and Ambiguous
        if pa.is_locality_unresolved:
            unresolved_records.append({
                "address_id": pa.address_id,
                "town_id": pa.town_id,
                "address_text": pa.address_text,
                "reason": "locality_unresolved",
                "locality_metadata": json.dumps(pa.locality_metadata),
                "other_details": pa.other_details,
            })

        if pa.is_ambiguous:
            ambiguous_records.append({
                "address_id": pa.address_id,
                "town_id": pa.town_id,
                "address_text": pa.address_text,
                "reason": "ambiguous_match",
                "locality_metadata": json.dumps(pa.locality_metadata),
                "other_details": pa.other_details,
            })

    weak_spans_df = pd.DataFrame(span_records)
    weak_labels_df = pd.DataFrame(token_records)
    bio_dataset_df = pd.DataFrame(bio_records)
    unresolved_df = pd.DataFrame(unresolved_records)
    ambiguous_df = pd.DataFrame(ambiguous_records)

    return weak_spans_df, weak_labels_df, bio_dataset_df, unresolved_df, ambiguous_df
