import re
from typing import Dict, Any, List, Optional
import pandas as pd
from pathlib import Path

from src import config


# Key landmarks and POI terms commonly referenced by field agents
REMARK_POI_TERMS = [
    # English & Romanized
    r"\bmilk\s+dairy\b",
    r"\bmilk\s+booth\b",
    r"\bdoodh\s+dairy\b",
    r"\bwater\s+tank\b",
    r"\bpaani\s+ki\s+tanki\b",
    r"\boverhead\s+tank\b",
    r"\btanki\b",
    r"\bpost\s+office\b",
    r"\bdakghar\b",
    r"\bchildren\s+park\b",
    r"\bpark\b",
    r"\bbus\s+stop\b",
    r"\bbus\s+stand\b",
    r"\bcommunity\s+hall\b",
    r"\bbarat\s+ghar\b",
    r"\bmedical\s+store\b",
    r"\bpharmacy\b",
    r"\bgovernment\s+school\b",
    r"\bgovt\s+school\b",
    r"\bschool\b",
    r"\bshaale\b",
    r"\bmosque\b",
    r"\bmasjid\b",
    r"\bmasidi\b",
    r"\bchurch\b",
    r"\b(?:hanuman|ganesh|ganapathi|shiva|sai\s+baba|om\s+sai|anjaneya|ram)?\s*(?:temple|mandir|gudi|devasthana)\b",
    r"\bsbi\s+bank\b",
    r"\bbank\b",
    r"\batm\b",
    r"\bhospital\b",
    r"\bration\s+shop\b",
    r"\bpds\s+shop\b",
    r"\bpetrol\s+bunk\b",
    r"\bpetrol\s+pump\b",
    r"\bcircle\b",
    r"\bjunction\b",
    # Indic scripts
    r"ಮಸೀದಿ",
    r"ದೇವಸ್ಥಾನ",
    r"ಗುಡಿ",
    r"ಚರ್ಚ್",
    r"ಶಾಲೆ",
    r"ಹಾಲು\s+ಡೈರಿ",
    r"ನೀರಿನ\s+ಟ್ಯಾಂಕ್",
    r"ಬಸ್\s+ನಿಲ್ದಾಣ",
    r"ಉದ್ಯಾನವನ",
]

REMARK_POI_REGEX = re.compile("|".join(REMARK_POI_TERMS), re.IGNORECASE)

# Directional & spatial offset markers
DIRECTIONAL_PATTERNS = [
    # Distance + cardinal/relative (e.g. "100m north", "50 meters ahead", "2 lanes ahead")
    r"\b\d+\s*(?:m|meter|meters|mtr|km)?\s*(?:north|south|east|west|ahead|back|further)\b",
    r"\b\d+\s*(?:lane|lanes|cross|crosses|gali|galis)\s*(?:ahead|munde|aage|baad)\b",
    # Relative spatial keywords (English, Kannada, Hindi)
    r"\b(?:near|opp\.?|opposite|behind|beside|next\s+to|adjacent(?:\s+to)?|in\s+front\s+of|close\s+to)\b",
    r"\b(?:hattira|hinde|hindhe|eduru|edurru|pakka|munde|mundhe)\b",
    r"\b(?:ke\s+paas|ke\s+pass|ke\s+samne|ke\s+saamne|ke\s+piche|ke\s+peechhe|ke\s+bagal(?:\s+mein)?|ke\s+aage)\b",
    r"\b(?:samne\s+wali\s+gali|eduru\s+road)\b",
]

DIRECTIONAL_REGEX = re.compile("|".join(DIRECTIONAL_PATTERNS), re.IGNORECASE)

# Outcome vs Remark contradiction signatures
# When outcome suggests success (met_borrower, met_family, cash_collected) but remark indicates failure/relocation
OUTCOME_SUCCESS = {"met_borrower", "met_family", "cash_collected"}
CONTRADICTION_KEYWORDS_FAILURE = [
    r"\b(?:locked|band|beega|beega\s+haakide|ghar\s+band)\b",
    r"\b(?:shifted|shift|khali|jaaga\s+bittidare)\b",
    r"\b(?:untraceable|sikkilla|not\s+found|nahi\s+mila|address\s+galat|address\s+tappu|galat\s+address)\b",
    r"\b(?:no\s+such\s+person|koi\s+nahi\s+rehta|yaaru\s+illa)\b",
]
CONTRADICTION_FAILURE_REGEX = re.compile("|".join(CONTRADICTION_KEYWORDS_FAILURE), re.IGNORECASE)

# When outcome is negative (address_not_traceable, no_such_person) but remark suggests positive contact
OUTCOME_FAILURE = {"address_not_traceable", "no_such_person"}
CONTRADICTION_KEYWORDS_SUCCESS = [
    r"\b(?:met\s+customer|customer\s+se\s+mila|customer\s+sikkidru|collected\s+cash|baat\s+hui)\b",
]
CONTRADICTION_SUCCESS_REGEX = re.compile("|".join(CONTRADICTION_KEYWORDS_SUCCESS), re.IGNORECASE)


class FieldVisitRemarkParser:
    """
    Parser for qualitative field visit remarks from field agents.
    Extracts:
    1. Landmarks & POIs mentioned in free-text remarks
    2. Directional & Spatial Offset vectors
    3. Outcome vs Remark Contradiction & Integrity flags
    4. Explicit address corrections (e.g. corrected house numbers, corrected lanes/landmarks)
    """

    def __init__(self):
        pass

    def extract_landmarks(self, remark: str) -> List[str]:
        """Extract spatial landmarks and points of interest mentioned in remarks."""
        if not remark or pd.isna(remark):
            return []

        landmarks = []
        # Pattern 1: Spatial preposition + POI (e.g. "behind Milk Dairy", "opposite Children Park", "near Temple")
        prep_poi_pattern = re.compile(
            r"(?i)\b(?:near|nr\.?|opp\.?|opposite|behind|beside|next\s+to|in\s+front\s+of)\s+([A-Za-z0-9\s]+?)(?:,|\.|\;|\bhinde\b|\beduru\b|\bhattira\b|\bide\b|\bhai\b|\bke\b|$)"
        )
        for m in prep_poi_pattern.finditer(remark):
            match_str = m.group(0).strip(" ,.;")
            # Filter noise
            if len(match_str.split()) >= 2 and any(p.search(match_str) for p in [REMARK_POI_REGEX]):
                landmarks.append(match_str)

        # Pattern 2: POI + transliterated suffix (e.g. "Milk Dairy hinde", "Ganapathi Gudi eduru", "Hanuman Mandir ke paas")
        suffix_poi_pattern = re.compile(
            r"(?i)([A-Za-z0-9\s]+?)\s+(?:hinde|hindhe|eduru|edurru|hattira|pakka|ke\s+paas|ke\s+saamne|ke\s+piche)"
        )
        for m in suffix_poi_pattern.finditer(remark):
            cand = m.group(0).strip(" ,.;")
            if REMARK_POI_REGEX.search(cand):
                landmarks.append(cand)

        # Pattern 3: Standalone POI keywords if not already captured
        for m in REMARK_POI_REGEX.finditer(remark):
            poi_str = m.group(0).strip()
            if not any(poi_str.lower() in lm.lower() for lm in landmarks):
                landmarks.append(poi_str)

        # Deduplicate while preserving order
        unique_lms = []
        for lm in landmarks:
            clean = lm.strip(" ,.;")
            if clean and clean not in unique_lms and len(clean) > 2:
                unique_lms.append(clean)

        return unique_lms

    def extract_directionals(self, remark: str) -> List[str]:
        """Extract directional vectors and spatial offsets (e.g. '100m north', '2 lanes ahead', 'eduru road')."""
        if not remark or pd.isna(remark):
            return []

        directionals = []
        for m in DIRECTIONAL_REGEX.finditer(remark):
            val = m.group(0).strip(" ,.;")
            if val and val not in directionals:
                directionals.append(val)
        return directionals

    def detect_contradiction(self, outcome: str, remark: str) -> Dict[str, Any]:
        """
        Detect logical contradictions between agent-selected outcome status
        and actual text recorded in remarks.
        """
        outcome_clean = str(outcome).strip().lower()
        remark_clean = str(remark).strip().lower()

        is_contradiction = False
        contradiction_reason = None

        if outcome_clean in OUTCOME_SUCCESS:
            fail_match = CONTRADICTION_FAILURE_REGEX.search(remark_clean)
            if fail_match:
                is_contradiction = True
                contradiction_reason = f"Outcome is '{outcome}' but remark indicates failure/closure/shift: '{fail_match.group(0)}'"

        elif outcome_clean in OUTCOME_FAILURE:
            succ_match = CONTRADICTION_SUCCESS_REGEX.search(remark_clean)
            if succ_match:
                is_contradiction = True
                contradiction_reason = f"Outcome is '{outcome}' but remark suggests contact: '{succ_match.group(0)}'"

        return {
            "is_contradiction": is_contradiction,
            "contradiction_reason": contradiction_reason,
        }

    def extract_address_corrections(self, remark: str) -> Dict[str, Optional[str]]:
        """
        Identify if agent supplied corrected house number, landmark, lane, or locality inside the remark.
        E.g. 'address wrong, house near Milk Booth', 'nija mane Milk Dairy hinde ide, 2 cross munde'
        """
        if not remark or pd.isna(remark):
            return {"has_correction": False, "corrected_details": None}

        correction_indicators = re.compile(
            r"(?i)\b(?:address\s+wrong|address\s+galat|address\s+tappu|nija\s+mane|actual\s+house|correct\s+address|ghar\s+yahan|naya\s+address)\b"
        )

        match = correction_indicators.search(remark)
        if match:
            # Extract clause after the correction indicator
            clause_after = remark[match.start():].strip(" ,.;")
            return {
                "has_correction": True,
                "corrected_details": clause_after,
            }

        return {"has_correction": False, "corrected_details": None}

    def parse_record(self, outcome: str, remark: str) -> Dict[str, Any]:
        """Parse complete remark record into structured insight dictionary."""
        lms = self.extract_landmarks(remark)
        dirs = self.extract_directionals(remark)
        contra = self.detect_contradiction(outcome, remark)
        corr = self.extract_address_corrections(remark)

        return {
            "outcome": outcome,
            "remark": remark,
            "extracted_landmarks": lms,
            "directional_vectors": dirs,
            "is_contradiction": contra["is_contradiction"],
            "contradiction_reason": contra["contradiction_reason"],
            "has_address_correction": corr["has_correction"],
            "corrected_address_clause": corr["corrected_details"],
        }


def parse_field_visits(
    csv_path: Path = config.FIELD_VISITS_CSV,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    """
    Batch processes field_visits.csv and augments it with remark intelligence columns.
    """
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit).copy()

    parser = FieldVisitRemarkParser()
    parsed_rows = []

    for _, row in df.iterrows():
        res = parser.parse_record(str(row.get("outcome", "")), str(row.get("remark", "")))
        parsed_rows.append({
            "visit_id": row.get("visit_id"),
            "address_id": row.get("address_id"),
            "outcome": res["outcome"],
            "remark": res["remark"],
            "extracted_landmarks": ", ".join(res["extracted_landmarks"]),
            "directional_vectors": ", ".join(res["directional_vectors"]),
            "is_contradiction": res["is_contradiction"],
            "contradiction_reason": res["contradiction_reason"],
            "has_address_correction": res["has_address_correction"],
            "corrected_address_clause": res["corrected_address_clause"],
        })

    return pd.DataFrame(parsed_rows)
