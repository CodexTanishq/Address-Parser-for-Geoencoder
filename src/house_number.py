import re
from typing import Optional, List, Tuple
from src import config
from src.utils import AddressSpan

# Prefixes that introduce a house number
# We explicitly forbid preceding street keywords so 'Gali No. 4' or 'Ward No. 3' are never matched
PREFIX_HNO_REGEX = re.compile(
    r"(?<!\bgali\s)(?<!\bward\s)(?<!\broad\s)(?<!\brd\s)(?<!\blane\s)(?<!\bln\s)(?<!\bstreet\s)(?<!\bst\s)"
    r"(?<!\bगली\s)"
    r"(?:(?:\b(?:h(?:ouse)?\.?\s*no\.?|hno|door\s*no\.?|d\.?\s*no\.?|flat(?:\s*no\.?)?|plot(?:\s*no\.?)?|house|no\.?))|#)\s*"
    r"([a-z0-9]+(?:[/-][a-z0-9]+)?)\b",
    re.IGNORECASE,
)

# Standalone slash numbers (e.g., 217/2, 367/3, 89/2, 313/2)
SLASH_HNO_REGEX = re.compile(
    r"\b(\d+[a-z]?[/-]\d+[a-z]?)\b",
    re.IGNORECASE,
)

# Standalone hyphenated plot/house numbers (e.g., A-123, 12-A)
HYPHEN_HNO_REGEX = re.compile(
    r"\b([a-z]-\d+|\d+-[a-z])\b",
    re.IGNORECASE,
)

# Ordinal pattern to reject
ORDINAL_REGEX = re.compile(r"^\d+(?:st|nd|rd|th)$", re.IGNORECASE)


def extract_house_number(
    text: str,
    protected_spans: Optional[List[AddressSpan]] = None,
) -> Tuple[Optional[str], Optional[AddressSpan]]:
    """
    Extract HOUSE_NUMBER entity from address text with high precision.
    Supports:
    - H.No. 356, H NO 356, HOUSE 52, HOUSE NO 52, NO. 173, NO 173, #81
    - 217/2, 313/2, 89/2, A-123, 12-A, Flat 304, Plot 21, Door No 45
    Strictly forbids:
    - Ordinals (e.g., 6th from '6th Cross', 5th from '5th Main')
    - Numbers that belong to Gali, Ward, Road, Lane, Street, or Pincode.
    """
    protected = protected_spans or []

    def is_protected(start: int, end: int) -> bool:
        return any(max(start, p.start) < min(end, p.end) for p in protected)

    # 1. Check prefixed house numbers (e.g., #81, H.No. 356, House 52, No. 173)
    for match in PREFIX_HNO_REGEX.finditer(text):
        val = match.group(1)
        # Check if ordinal (e.g. '6th')
        if ORDINAL_REGEX.match(val):
            continue
        # Check if 6-digit pincode
        if len(val) == 6 and val.isdigit():
            continue

        full_start, full_end = match.span()
        # Find the span of just the value
        val_start = match.start(1)
        val_end = match.end(1)

        if not is_protected(val_start, val_end):
            span = AddressSpan(
                entity_type="HOUSE_NUMBER",
                text=val,
                start=val_start,
                end=val_end,
                confidence=config.CONF_HIGH,
                source="regex_prefixed",
                match_score=1.0,
                metadata={"full_match": match.group(0), "prefix": text[full_start:val_start].strip()},
            )
            return val, span

    # 2. Check standalone slash format (e.g., 217/2, 367/3, 89/2)
    for match in SLASH_HNO_REGEX.finditer(text):
        val = match.group(1)
        start, end = match.span()
        if not is_protected(start, end):
            span = AddressSpan(
                entity_type="HOUSE_NUMBER",
                text=val,
                start=start,
                end=end,
                confidence=config.CONF_HIGH,
                source="regex_slash",
                match_score=1.0,
                metadata={"format": "slash"},
            )
            return val, span

    # 3. Check standalone hyphenated alphanumeric format (e.g., A-123, 12-A)
    for match in HYPHEN_HNO_REGEX.finditer(text):
        val = match.group(1)
        start, end = match.span()
        if not is_protected(start, end):
            span = AddressSpan(
                entity_type="HOUSE_NUMBER",
                text=val,
                start=start,
                end=end,
                confidence=config.CONF_HIGH,
                source="regex_hyphen",
                match_score=1.0,
                metadata={"format": "hyphen"},
            )
            return val, span

    return None, None
