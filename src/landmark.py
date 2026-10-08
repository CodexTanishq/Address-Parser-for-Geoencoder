import re
from typing import List, Tuple, Optional
from src import config
from src.utils import AddressSpan

# Prefix indicators (e.g., near, nr., opp, behind, beside, close to, next to, adjacent)
PREFIX_INDICATORS = re.compile(
    r"(?i)\b(?:near|nr\.?|opp\.?|opposite|behind|beside|close\s+to|next\s+to|adjacent(?:\s+to)?|adj\.?)\b\s*",
)

# Suffix indicators (multilingual and transliterated)
SUFFIX_INDICATORS = re.compile(
    r"(?i)\s*\b(?:hattira|hinde|hindhe|eduru|edurru|pakka|ಹತ್ತಿರ|ಹಿಂದೆ|ಎದುರು|ಪಕ್ಕ|"
    r"ke\s+paas|ke\s+pass|ke\s+samne|ke\s+saamne|ke\s+bagal(?:\s+mein|\s+me)?|ke\s+piche|ke\s+peechhe|"
    r"के\s+पास|के\s+सामने|के\s+पीछे|के\s+बगल(?:\s+में)?)\b",
)

# Standalone POI keywords (English, Transliterated, Indic)
STANDALONE_POIS = [
    # General / Commercial / Facilities
    r"\bmilk\s+dairy\b",
    r"\bhalina\s+dairi\b",
    r"\bhaalina\s+dairy\b",
    r"\bpds\s+shop\b",
    r"\bration\s+shop\b",
    r"\bnyayabele\s+angadi\b",
    r"\bgovernment\s+school\b",
    r"\bgovt\s+school\b",
    r"\bsarkari\s+shaale\b",
    r"\bsarakari\s+skula\b",
    r"\bwater\s+tank\b",
    r"\boverhead\s+tank\b",
    r"\btanki\b",
    r"\bpaani\s+ki\s+tanki\b",
    r"\bmedical\s+store\b",
    r"\bpharmacy\b",
    r"\bdawai\s+ki\s+dukaan\b",
    r"\bdawai\s+ki\s+dukan\b",
    r"\bchildren\s+park\b",
    r"\bbus\s+stand\b",
    r"\bbus\s+stop\b",
    r"\bbus\s+adda\b",
    r"\brailway\s+station\b",
    r"\bpost\s+office\b",
    r"\bdakghar\b",
    r"\bpetrol\s+bunk\b",
    r"\bpetrol\s+pump\b",
    r"\bkalyana\s+mantapa\b",
    r"\bkalyana\s+mamtapa\b",
    r"\bbarat\s+ghar\b",
    r"\bcommunity\s+hall\b",
    r"\bsbi\s+bank\b",
    r"\bcanara\s+bank\b",
    r"\bhdfc\s+bank\b",
    r"\bicici\s+bank\b",
    r"\bbank\b",
    r"\batm\b",
    r"\bhospital\b",
    r"\bclinic\b",
    r"\bmarket\b",
    r"\bcircle\b",
    r"\bjunction\b",
    r"\bmilk\s+booth\b",
    # Religious POIs
    r"\b(?:hanuman|ganesh|ganapathi|shiva|sai\s+baba|om\s+sai|anjaneya|ram)?\s*(?:temple|mandir|gudi|devasthana)\b",
    r"\bchurch\b",
    r"\bmosque\b",
    r"\bmasjid\b",
    # Indic script POIs
    r"ಚರ್ಚ್",
    r"ಕಲ್ಯಾಣ\s+ಮಂಟಪ",
    r"ಬಸ್\s+ನಿಲ್ದಾಣ",
    r"ನ್ಯಾಯಬೆಲೆ\s+ಅಂಗಡಿ",
    r"ಸರ್ಕಾರಿ\s+ಶಾಲೆ",
    r"ಗಣಪತಿ\s+ದೇವಸ್ಥಾನ",
    r"ದೇವಸ್ಥಾನ",
    r"ಗುಡಿ",
    r"ಶಾಲೆ",
    r"ಮಸೀದಿ",
    r"ಮೆಡಿಕಲ್\s+ಶಾಪ್",
    r"ಹಾಲಿನ\s+ಡೈರಿ",
    r"ಉದ್ಯಾನವನ",
    r"ಪೆಟ್ರೋಲ್\s+ಬಂಕ್",
    r"ನೀರಿನ\s+ಟ್ಯಾಂಕ್",
    r"मस्जिद",
    r"मंदिर",
    r"राशन\s+की\s+दुकान",
    r"सरकारी\s+स्कूल",
    r"डाकघर",
    r"दवाई\s+की\s+दुकान",
    r"बारात\s+घर",
    r"बस\s+स्टॉप",
    r"पानी\s+की\s+टंकी",
    r"हनुमान\s+मंदिर",
    r"गणेश\s+मंदिर",
    r"दूध\s+डेयरी",
]

POI_PATTERN = re.compile("|".join(STANDALONE_POIS), re.IGNORECASE)


def get_unallocated_segments(
    text: str, protected_spans: List[AddressSpan]
) -> List[Tuple[int, int, str]]:
    """
    Get all text segments that are NOT claimed by protected entities
    (Pincode, Town, Locality, Ward, House Number, Street).
    """
    if not protected_spans:
        return [(0, len(text), text)]

    sorted_spans = sorted(protected_spans, key=lambda s: s.start)
    segments = []
    curr = 0

    for span in sorted_spans:
        if span.start > curr:
            seg_text = text[curr:span.start]
            segments.append((curr, span.start, seg_text))
        curr = max(curr, span.end)

    if curr < len(text):
        seg_text = text[curr:len(text)]
        segments.append((curr, len(text), seg_text))

    return segments


def extract_landmarks(
    text: str, protected_spans: Optional[List[AddressSpan]] = None
) -> List[AddressSpan]:
    """
    Extract LANDMARK entities strictly from unallocated candidate text segments.
    This guarantees that landmarks NEVER absorb house numbers, streets, wards,
    localities, towns, or pincodes.
    """
    protected = protected_spans or []
    segments = get_unallocated_segments(text, protected)
    landmark_spans: List[AddressSpan] = []

    for seg_start, seg_end, seg_text in segments:
        # Clean segment boundaries (remove leading/trailing commas, dashes, whitespace)
        # We split by commas or semicolons within an unallocated segment to avoid bridging unrelated clauses
        clause_matches = list(re.finditer(r"[^,;\-]+", seg_text))

        for clause_m in clause_matches:
            clause = clause_m.group(0).strip()
            if not clause:
                continue

            c_start = seg_start + clause_m.start() + (len(clause_m.group(0)) - len(clause_m.group(0).lstrip()))
            c_end = c_start + len(clause)

            # Check Pattern 1: Prefix indicator (e.g., nr church, opp water tank)
            p_match = PREFIX_INDICATORS.search(clause)
            if p_match and p_match.start() == 0:
                # The entire clause starting with the prefix indicator is a landmark
                landmark_spans.append(
                    AddressSpan(
                        entity_type="LANDMARK",
                        text=clause,
                        start=c_start,
                        end=c_end,
                        confidence=config.CONF_HIGH,
                        source="context_prefix",
                        match_score=1.0,
                        metadata={"indicator": p_match.group(0).strip()},
                    )
                )
                continue

            # Check Pattern 2: Suffix indicator (e.g., Ganapathi Gudi hattira, Masjid ke paas)
            s_match = SUFFIX_INDICATORS.search(clause)
            if s_match and s_match.end() == len(clause):
                landmark_spans.append(
                    AddressSpan(
                        entity_type="LANDMARK",
                        text=clause,
                        start=c_start,
                        end=c_end,
                        confidence=config.CONF_HIGH,
                        source="context_suffix",
                        match_score=1.0,
                        metadata={"indicator": s_match.group(0).strip()},
                    )
                )
                continue

            # Check Pattern 3: Standalone POI match within clause
            poi_match = POI_PATTERN.search(clause)
            if poi_match:
                # If the clause contains a known POI, extract the POI or entire meaningful clause
                # E.g., 'Milk Dairy', 'PDS Shop', 'Government School'
                landmark_spans.append(
                    AddressSpan(
                        entity_type="LANDMARK",
                        text=clause,
                        start=c_start,
                        end=c_end,
                        confidence=config.CONF_HIGH,
                        source="known_poi",
                        match_score=1.0,
                        metadata={"poi_keyword": poi_match.group(0)},
                    )
                )

    return landmark_spans
