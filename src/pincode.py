import re
from typing import Optional, Tuple, Set
from src import config
from src.utils import AddressSpan

# Exactly 6 digits, strictly isolated from adjacent digits
PINCODE_REGEX = re.compile(r"(?<!\d)(\d{6})(?!\d)")


def extract_pincode(
    text: str,
    town_id: Optional[str] = None,
    valid_pincodes_for_town: Optional[Set[str]] = None,
) -> Tuple[Optional[str], Optional[AddressSpan]]:
    """
    Extract a valid 6-digit pincode from address text.
    Prioritizes precision:
    - Does NOT match 5-digit, 7-digit, or fractional numbers (e.g., 96010, 1234567, 217/2, 6th).
    - If multiple conflicting pincodes are found, marks confidence as AMBIGUOUS.
    - If pincode agrees with town gazetteer, confidence is HIGH.
    """
    matches = list(PINCODE_REGEX.finditer(text))

    if not matches:
        return None, None

    # Check for multiple distinct pincodes
    unique_pins = {m.group(1) for m in matches}
    if len(unique_pins) > 1:
        # Ambiguous case: multiple conflicting 6-digit numbers found
        first_m = matches[0]
        span = AddressSpan(
            entity_type="PINCODE",
            text=first_m.group(1),
            start=first_m.start(),
            end=first_m.end(),
            confidence=config.CONF_AMBIGUOUS,
            source="pincode_regex",
            match_score=0.5,
            metadata={"found_pincodes": list(unique_pins)},
        )
        return first_m.group(1), span

    # Exactly one unique 6-digit pincode found
    match = matches[0]
    pincode_val = match.group(1)

    # First digit of valid Indian pincodes is 1-9
    if pincode_val[0] == "0":
        return None, None

    confidence = config.CONF_HIGH
    match_score = 1.0

    # Cross-reference with town gazetteer if available
    if valid_pincodes_for_town is not None:
        if pincode_val in valid_pincodes_for_town:
            confidence = config.CONF_HIGH
        else:
            # Pincode exists but disagrees with town's gazetteer
            confidence = config.CONF_MEDIUM
            match_score = 0.8

    span = AddressSpan(
        entity_type="PINCODE",
        text=pincode_val,
        start=match.start(),
        end=match.end(),
        confidence=confidence,
        source="pincode_regex",
        match_score=match_score,
        metadata={"town_id": town_id},
    )

    return pincode_val, span
