import re
from typing import List
from src import config
from src.utils import AddressSpan

# Street regex patterns
STREET_PATTERNS = [
    # Ordinal / Number + Cross/Cr/Crs/X (e.g., 6th Cross, 7th Cr, 9th crs, 6th X, 1st x)
    re.compile(
        r"\b(?:\d+(?:st|nd|rd|th)?)\s*(?:cross|crs|cr|x)\b",
        re.IGNORECASE,
    ),
    # Ordinal / Number + Main/Mn (e.g., 5th Main, 1st main, 8th mn, 2nd mn)
    re.compile(
        r"\b(?:\d+(?:st|nd|rd|th)?)\s*(?:main|mn)\b",
        re.IGNORECASE,
    ),
    # Ordinal / Number + Road/Rd (e.g., 3rd Road, 2nd road, 5th Road)
    re.compile(
        r"\b(?:\d+(?:st|nd|rd|th)?)\s*(?:road|rd)\b",
        re.IGNORECASE,
    ),
    # Road/Rd + Number (e.g., Rd 6, Road 5, rd 2, rd. 4)
    re.compile(
        r"\b(?:road|rd)\.?\s*(?:no\.?\s*)?(\d+)\b",
        re.IGNORECASE,
    ),
    # Gali / गली + Number (e.g., Gali 4, Gali No. 5, Gali no-4, gali 10, गली नं. 3)
    re.compile(
        r"\b(?:gali|गली)\s*(?:no\.?|no-|नं\.?)?\s*(\d+)\b",
        re.IGNORECASE,
    ),
    # Ordinal / Number + Lane/Ln (e.g., 2nd Lane, 1st ln)
    re.compile(
        r"\b(?:\d+(?:st|nd|rd|th)?)\s*(?:lane|ln)\b",
        re.IGNORECASE,
    ),
    # Lane/Ln + Number (e.g., Lane 2)
    re.compile(
        r"\b(?:lane|ln)\.?\s*(?:no\.?\s*)?(\d+)\b",
        re.IGNORECASE,
    ),
    # Ordinal / Number + Street/St (e.g., 1st Street, 5th st)
    re.compile(
        r"\b(?:\d+(?:st|nd|rd|th)?)\s*(?:street|st)\b",
        re.IGNORECASE,
    ),
]


def extract_streets(text: str) -> List[AddressSpan]:
    """
    Extract individual STREET spans from address text.
    Handles cross, main, road, gali, lane, street abbreviations and ordinals.
    Unrelated streets (e.g., '6th Cross, 5th Main') are returned as separate spans.
    """
    spans: List[AddressSpan] = []

    for pattern in STREET_PATTERNS:
        for match in pattern.finditer(text):
            start = match.start()
            end = match.end()

            # Avoid adding an overlapping street span
            is_overlapping = any(
                max(start, existing.start) < min(end, existing.end)
                for existing in spans
            )
            if not is_overlapping:
                span = AddressSpan(
                    entity_type="STREET",
                    text=match.group(0),
                    start=start,
                    end=end,
                    confidence=config.CONF_HIGH,
                    source="street_regex",
                    match_score=1.0,
                    metadata={"matched_text": match.group(0)},
                )
                spans.append(span)

    # Sort spans by starting position in the address text
    spans.sort(key=lambda s: s.start)
    return spans
