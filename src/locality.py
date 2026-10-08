from typing import Optional, Dict, List, Tuple, Any
from difflib import SequenceMatcher
import re
import pandas as pd
from src import config
from src.utils import AddressSpan
from src.normalization import normalize_for_matching


class LocalityResolver:
    """
    High-precision Locality resolution using the gazetteer.
    Exploits town_id, pincode, exact phrase matching, n-gram candidate spans,
    and fuzzy similarity with margin checking to prevent ambiguity.
    """

    def __init__(self, localities_df: pd.DataFrame):
        self.localities_df = localities_df.copy()
        # Group localities by town_id
        self.town_localities: Dict[str, List[Dict[str, Any]]] = {}
        for _, row in localities_df.iterrows():
            tid = str(row["town_id"])
            loc_entry = {
                "locality_id": str(row["locality_id"]),
                "locality_name": str(row["locality_name"]),
                "town_id": tid,
                "pincode": str(row["pincode"]),
                "norm_name": normalize_for_matching(str(row["locality_name"])),
            }
            if tid not in self.town_localities:
                self.town_localities[tid] = []
            self.town_localities[tid].append(loc_entry)

    def extract_candidates(
        self, text: str, max_words: int = 4
    ) -> List[Tuple[str, int, int]]:
        """
        Extract 1-word, 2-word, 3-word, and 4-word spans with their exact character offsets.
        Ignores punctuation-only tokens.
        """
        # Find word tokens with spans
        word_matches = list(re.finditer(r"\b[\w'-]+\b", text))
        candidates = []
        n = len(word_matches)

        for length in range(1, min(max_words + 1, n + 1)):
            for i in range(n - length + 1):
                start = word_matches[i].start()
                end = word_matches[i + length - 1].end()
                phrase = text[start:end]
                candidates.append((phrase, start, end))

        return candidates

    def resolve(
        self,
        text: str,
        town_id: str,
        pincode: Optional[str] = None,
    ) -> Tuple[Optional[str], Optional[AddressSpan], Dict[str, Any]]:
        """
        Resolve locality for the given address text within the specified town.
        Returns (canonical_locality_name, span, metadata).
        """
        metadata = {
            "locality_id": None,
            "locality_name": None,
            "matched_phrase": None,
            "match_score": 0.0,
            "second_best_score": 0.0,
            "score_margin": 0.0,
            "match_type": "none",
            "confidence": config.CONF_LOW,
        }

        # Filter gazetteer entries by authoritative town_id
        gazetteer_entries = self.town_localities.get(town_id, [])
        if not gazetteer_entries:
            return None, None, metadata

        candidates = self.extract_candidates(text, max_words=4)
        if not candidates:
            return None, None, metadata

        # Stage 1: Exact normalized match
        # Try longer phrases first for maximum specificity
        for phrase, start, end in sorted(candidates, key=lambda x: len(x[0]), reverse=True):
            norm_phrase = normalize_for_matching(phrase)
            for entry in gazetteer_entries:
                if norm_phrase == entry["norm_name"]:
                    metadata.update({
                        "locality_id": entry["locality_id"],
                        "locality_name": entry["locality_name"],
                        "matched_phrase": phrase,
                        "match_score": 1.0,
                        "second_best_score": 0.0,
                        "score_margin": 1.0,
                        "match_type": "exact_normalized",
                        "confidence": config.CONF_HIGH,
                    })
                    span = AddressSpan(
                        entity_type="LOCALITY",
                        text=phrase,
                        start=start,
                        end=end,
                        confidence=config.CONF_HIGH,
                        source="exact_gazetteer",
                        match_score=1.0,
                        metadata=metadata,
                    )
                    return entry["locality_name"], span, metadata

        # Stage 2: Fuzzy phrase matching
        scored_matches = []
        for phrase, start, end in candidates:
            norm_phrase = normalize_for_matching(phrase)
            # Skip tiny phrases that are too generic
            if len(norm_phrase) < 4:
                continue

            for entry in gazetteer_entries:
                sim = SequenceMatcher(None, norm_phrase, entry["norm_name"]).ratio()
                # If word tokens match after abbreviation expansion
                phrase_words = set(norm_phrase.split())
                entry_words = set(entry["norm_name"].split())
                if phrase_words == entry_words:
                    sim = max(sim, 0.98)
                elif phrase_words.issubset(entry_words) and len(phrase_words) >= 2:
                    sim = max(sim, 0.88)

                if sim >= 0.75:
                    scored_matches.append((sim, entry, phrase, start, end))

        if not scored_matches:
            return None, None, metadata

        # Sort by score descending
        scored_matches.sort(key=lambda x: x[0], reverse=True)

        best_score, best_entry, best_phrase, best_start, best_end = scored_matches[0]
        second_best_score = 0.0

        # Find second best score for a different locality
        for sim, entry, _, _, _ in scored_matches[1:]:
            if entry["locality_id"] != best_entry["locality_id"]:
                second_best_score = sim
                break

        score_margin = best_score - second_best_score

        metadata.update({
            "locality_id": best_entry["locality_id"],
            "locality_name": best_entry["locality_name"],
            "matched_phrase": best_phrase,
            "match_score": round(best_score, 3),
            "second_best_score": round(second_best_score, 3),
            "score_margin": round(score_margin, 3),
            "match_type": "fuzzy_gazetteer",
        })

        # Ambiguity and confidence check
        if best_score >= 0.85:
            if second_best_score > 0 and score_margin < 0.12:
                # Ambiguous: top candidate not sufficiently better than second candidate
                metadata["confidence"] = config.CONF_AMBIGUOUS
                span = AddressSpan(
                    entity_type="LOCALITY",
                    text=best_phrase,
                    start=best_start,
                    end=best_end,
                    confidence=config.CONF_AMBIGUOUS,
                    source="fuzzy_gazetteer",
                    match_score=round(best_score, 3),
                    metadata=metadata,
                )
                return best_entry["locality_name"], span, metadata
            else:
                metadata["confidence"] = config.CONF_HIGH
                span = AddressSpan(
                    entity_type="LOCALITY",
                    text=best_phrase,
                    start=best_start,
                    end=best_end,
                    confidence=config.CONF_HIGH,
                    source="fuzzy_gazetteer",
                    match_score=round(best_score, 3),
                    metadata=metadata,
                )
                return best_entry["locality_name"], span, metadata

        # Below 0.85: leave unlabeled (low confidence)
        metadata["confidence"] = config.CONF_LOW
        return None, None, metadata
