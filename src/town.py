import re
from typing import Optional, Dict, Tuple, List
import pandas as pd
from src import config
from src.utils import AddressSpan

# Regex patterns for finding town mentions in address text
TOWN_PATTERNS = {
    "T1": [re.compile(r"\bkaveripura\b", re.IGNORECASE)],
    "T2": [
        re.compile(r"\bdevgarh\s+nagar\b", re.IGNORECASE),
        re.compile(r"\bdevgarh\s+ngr\b", re.IGNORECASE),
        re.compile(r"\bdevgarh\s+nagr\b", re.IGNORECASE),
        re.compile(r"\bdevgarh\b", re.IGNORECASE),
    ],
    "T3": [re.compile(r"\bnavanagara\s+east\b", re.IGNORECASE)],
}


class TownResolver:
    """Authoritative town resolver using town_id and gazetteer lookup."""

    def __init__(self, towns_df: pd.DataFrame):
        self.town_map: Dict[str, str] = {}
        for _, row in towns_df.iterrows():
            self.town_map[str(row["town_id"])] = str(row["town_name"])

    def get_canonical_name(self, town_id: str) -> Optional[str]:
        """Return the authoritative town name for a town_id."""
        return self.town_map.get(town_id)

    def extract_town_span(
        self, text: str, town_id: str
    ) -> Tuple[Optional[str], Optional[AddressSpan]]:
        """
        Extract the exact character span of the town in the address text.
        If town name is present in address text, returns the span with HIGH confidence.
        If not present in text, returns (canonical_name, None) without inventing a fake span.
        """
        canonical_name = self.get_canonical_name(town_id)
        if not canonical_name:
            # E.g. OUT or unknown
            return None, None

        patterns = TOWN_PATTERNS.get(town_id, [])
        for pat in patterns:
            match = pat.search(text)
            if match:
                span = AddressSpan(
                    entity_type="TOWN",
                    text=match.group(0),
                    start=match.start(),
                    end=match.end(),
                    confidence=config.CONF_HIGH,
                    source="town_lookup",
                    match_score=1.0,
                    metadata={"town_id": town_id, "canonical_name": canonical_name},
                )
                return canonical_name, span

        # Town is known from town_id, but not explicitly present in address string
        return canonical_name, None
