from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import pandas as pd

from src import config
from src.utils import AddressSpan
from src.transliteration import detect_scripts, transliterate_with_mapping
from src.normalization import normalize_text
from src.pincode import extract_pincode
from src.town import TownResolver
from src.locality import LocalityResolver
from src.ward import extract_wards
from src.street import extract_streets
from src.house_number import extract_house_number
from src.landmark import extract_landmarks, get_unallocated_segments


@dataclass
class ParsedAddress:
    """Complete structured parsing result for an address."""
    address_id: str
    town_id: str
    address_text: str
    address_transliterated: str
    address_normalized: str
    scripts: List[str]
    pincode: Optional[str] = None
    town_name: Optional[str] = None
    locality_name: Optional[str] = None
    house_number: Optional[str] = None
    streets: List[str] = field(default_factory=list)
    wards: List[str] = field(default_factory=list)
    landmarks: List[str] = field(default_factory=list)
    other_details: str = ""
    spans: List[AddressSpan] = field(default_factory=list)
    locality_metadata: Dict[str, Any] = field(default_factory=dict)
    overall_confidence: str = config.CONF_HIGH
    is_ambiguous: bool = False
    is_locality_unresolved: bool = False


class AddressParser:
    """
    Modular deterministic address parser acting as the high-precision teacher
    for weak supervision. Prioritizes precision over recall.
    """

    def __init__(self, towns_df: pd.DataFrame, localities_df: pd.DataFrame):
        self.towns_df = towns_df
        self.localities_df = localities_df
        self.town_resolver = TownResolver(towns_df)
        self.locality_resolver = LocalityResolver(localities_df)

        # Build set of valid pincodes per town
        self.town_valid_pincodes: Dict[str, set] = {}
        for _, row in localities_df.iterrows():
            tid = str(row["town_id"])
            if tid not in self.town_valid_pincodes:
                self.town_valid_pincodes[tid] = set()
            self.town_valid_pincodes[tid].add(str(row["pincode"]))

    def parse(self, address_id: str, address_text: str, town_id: str) -> ParsedAddress:
        """
        Parse raw address into structured entities, weak spans, and other_details.
        """
        # Step 1: Script Detection & Transliteration
        scripts = detect_scripts(address_text)
        address_transliterated, _ = transliterate_with_mapping(address_text)
        address_normalized = normalize_text(address_text)

        # Step 2: Pincode Extraction
        valid_pins = self.town_valid_pincodes.get(town_id)
        pincode_val, pincode_span = extract_pincode(
            address_text, town_id=town_id, valid_pincodes_for_town=valid_pins
        )

        # Step 3: Town Resolution
        canonical_town, town_span = self.town_resolver.extract_town_span(
            address_text, town_id
        )

        # Step 4: Locality Resolution
        canonical_loc, loc_span, loc_meta = self.locality_resolver.resolve(
            address_text, town_id, pincode=pincode_val
        )
        is_loc_unresolved = (loc_span is None)
        is_ambiguous = (loc_meta.get("confidence") == config.CONF_AMBIGUOUS)

        # Step 5: Ward Extraction
        ward_spans = extract_wards(address_text)

        # Step 6: Street Extraction
        street_spans = extract_streets(address_text)

        # Protected spans so far
        pre_hno_protected = [s for s in [pincode_span, town_span, loc_span] if s]
        pre_hno_protected.extend(ward_spans)
        pre_hno_protected.extend(street_spans)

        # Step 7: House Number Extraction
        hno_val, hno_span = extract_house_number(
            address_text, protected_spans=pre_hno_protected
        )

        # Step 8: Conflict Resolution among structured entities
        all_structured_spans: List[AddressSpan] = []
        for s in [pincode_span, town_span, loc_span, hno_span]:
            if s:
                all_structured_spans.append(s)
        all_structured_spans.extend(ward_spans)
        all_structured_spans.extend(street_spans)

        # Resolve conflicts by precedence order
        all_structured_spans.sort(
            key=lambda s: config.PRECEDENCE_ORDER.index(s.entity_type)
            if s.entity_type in config.PRECEDENCE_ORDER
            else 99
        )
        resolved_structured: List[AddressSpan] = []
        for span in all_structured_spans:
            if not any(span.overlaps_with(existing) for existing in resolved_structured):
                resolved_structured.append(span)

        # Step 9: Landmark Extraction (strictly from unallocated text)
        landmark_spans = extract_landmarks(
            address_text, protected_spans=resolved_structured
        )

        # Combine all final spans and sort by start index
        final_spans = resolved_structured + landmark_spans
        final_spans.sort(key=lambda s: s.start)

        # Step 10: Extract other_details (unrecognized text)
        unalloc_segments = get_unallocated_segments(address_text, final_spans)
        other_parts = []
        for _, _, seg_text in unalloc_segments:
            # Clean up residual punctuation and spaces
            cleaned = seg_text.strip(" ,-;/")
            if cleaned:
                other_parts.append(cleaned)
        other_details = ", ".join(other_parts)

        # Determine overall confidence
        has_ambiguous_span = any(s.confidence == config.CONF_AMBIGUOUS for s in final_spans)
        if has_ambiguous_span or is_ambiguous:
            overall_confidence = config.CONF_AMBIGUOUS
            is_ambiguous = True
        elif is_loc_unresolved:
            overall_confidence = config.CONF_MEDIUM
        elif any(s.confidence == config.CONF_MEDIUM for s in final_spans):
            overall_confidence = config.CONF_MEDIUM
        else:
            overall_confidence = config.CONF_HIGH

        return ParsedAddress(
            address_id=address_id,
            town_id=town_id,
            address_text=address_text,
            address_transliterated=address_transliterated,
            address_normalized=address_normalized,
            scripts=scripts,
            pincode=pincode_val,
            town_name=canonical_town,
            locality_name=canonical_loc,
            house_number=hno_val,
            streets=[s.text for s in street_spans if s in final_spans],
            wards=[s.text for s in ward_spans if s in final_spans],
            landmarks=[s.text for s in landmark_spans],
            other_details=other_details,
            spans=final_spans,
            locality_metadata=loc_meta,
            overall_confidence=overall_confidence,
            is_ambiguous=is_ambiguous,
            is_locality_unresolved=is_loc_unresolved,
        )
