import re
from typing import List, Optional, Tuple
from src import config
from src.utils import AddressSpan

# Regex for Ward numbers
WARD_REGEX = re.compile(
    r"\bward\s*(?:no\.?|num\.?|number)?\s*(\d+)\b",
    re.IGNORECASE,
)


def extract_wards(text: str) -> List[AddressSpan]:
    """
    Extract WARD entities from address text.
    Matches forms like:
    - Ward 9
    - Ward No. 9
    - Ward Number 9
    - ward 3
    Ensures ward numbers are NOT confused with house numbers.
    """
    spans = []
    for match in WARD_REGEX.finditer(text):
        ward_num = match.group(1)
        span = AddressSpan(
            entity_type="WARD",
            text=match.group(0),
            start=match.start(),
            end=match.end(),
            confidence=config.CONF_HIGH,
            source="ward_regex",
            match_score=1.0,
            metadata={"ward_number": ward_num},
        )
        spans.append(span)
    return spans
