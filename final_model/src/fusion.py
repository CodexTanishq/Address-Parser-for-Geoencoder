"""
Milestone 4: Rule-Based Auditable Fusion Engine & Spatial Coordinate Resolver
Landmark rule (v3):
  1. Town is resolved first.
  2. Every POI of the detected landmark type in that town is collected.
  3. If the locality is known AND has POIs of that type, keep only those.
     Otherwise keep every POI of that type in the town.
  4. Every kept POI is returned with its own coordinates.
"""
import re
from typing import Dict, List, Any, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from final_model.src import config
from final_model.src.audit import build_spatial_poi_index
from final_model.src.labeller import (
    preprocess_address,
    LANDMARK_TYPE_LEXICON,
    RELATION_PATTERNS,
    LOCALITY_NAME_GUARDS,
)

ABBREVIATION_MAP = {
    'lyt': 'layout',
    'layt': 'layout',
    'layou': 'layout',
    'ngr': 'nagar',
    'nagr': 'nagar',
    'ngar': 'nagar',
    'colny': 'colony',
    'clny': 'colony',
    'col': 'colony',
    'blk': 'block',
    'rd': 'road',
    'mn': 'main',
    'x': 'cross',
    'crs': 'cross',
    'cr': 'cross',
    'nr': 'near',
    'opp': 'opposite',
    'bhd': 'behind',
}


def levenshtein_distance(s1: str, s2: str) -> int:
    """Computes Levenshtein edit distance between two strings."""
    if len(s1) < len(s2):
        s1, s2 = s2, s1
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (len(s2) + 1)
        for j, c2 in enumerate(s2):
            cost = 0 if c1 == c2 else 1
            curr[j + 1] = min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost)
        prev = curr
    return prev[len(s2)]


def fuzzy_word_match(word1: str, word2: str) -> bool:
    """
    Length rule:
    - max length <= 4: exact match only
    - max length 5-7: edit distance <= 1
    - max length >= 8: edit distance <= 2
    Adjacent letter swap (e.g. 'idnira' -> 'indira') counts as one edit.
    """
    if word1 == word2:
        return True
    l1, l2 = len(word1), len(word2)
    max_len = max(l1, l2)
    if max_len <= 4:
        return False
    if abs(l1 - l2) > 2:
        return False

    # Adjacent swap check (same length only)
    if l1 == l2 and max_len >= 5:
        diffs = [i for i in range(l1) if word1[i] != word2[i]]
        if len(diffs) == 2 and diffs[1] == diffs[0] + 1:
            if word1[diffs[0]] == word2[diffs[1]] and word1[diffs[1]] == word2[diffs[0]]:
                return True

    limit = 1 if max_len <= 7 else 2
    return levenshtein_distance(word1, word2) <= limit


def normalize_tokens(text: str) -> List[str]:
    """Lowercases, strips punctuation, applies the abbreviation map."""
    raw_tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
    return [ABBREVIATION_MAP.get(t, t) for t in raw_tokens]


def fuzzy_ngram_match(cand_tokens: List[str], target_tokens: List[str]) -> bool:
    """Matches a whole n-gram token by token using the length rule."""
    if len(cand_tokens) != len(target_tokens):
        return False
    for c_tok, t_tok in zip(cand_tokens, target_tokens):
        if not fuzzy_word_match(c_tok, t_tok):
            return False
    return True


class AddressFusionEngine:
    """Auditable multi-cue fusion geocoder and evidence resolver."""

    def __init__(self):
        self.towns_df = pd.read_csv(config.TOWNS_CSV)
        self.localities_df = pd.read_csv(config.LOCALITIES_CSV)
        self.landmarks_df = pd.read_csv(config.LANDMARKS_POI_CSV)

        # POI table with each POI's locality assignment (one row per POI)
        self.spatial_poi_df = build_spatial_poi_index(self.localities_df, self.landmarks_df)

        self.town_id_to_name = dict(zip(self.towns_df['town_id'], self.towns_df['town_name']))
        self.town_name_to_id = {v.lower(): k for k, v in self.town_id_to_name.items()}

        self.locality_records = self.localities_df.to_dict(orient='records')
        self.loc_to_towns = {}
        for r in self.locality_records:
            lname = str(r['locality_name']).lower().strip()
            self.loc_to_towns.setdefault(lname, []).append(r['town_id'])

        from final_model.src.labeller import DeterministicLabeller
        self.labeller = DeterministicLabeller()

        self.town_tfidf = joblib.load(config.TFIDF_TOWN_MODEL_PATH)
        self.loc_tfidf = joblib.load(config.TFIDF_LOCALITY_MODEL_PATH)

    def resolve_town(self, text: str, norm_raw: str, pincode: Optional[str], use_pincode: bool = True) -> Tuple[str, float, bool]:
        """
        Town resolution with pincode as a cue, not a rule.
        Returns: (town_id, confidence, conflict_flag)
        """
        low = norm_raw.lower()
        if (use_pincode and pincode and pincode.startswith('99')) or low.startswith('village') or ' taluk ' in low or ' tehsil ' in low:
            return 'OUT', 1.0, False

        text_tokens = normalize_tokens(text)
        found_text_towns = []
        for tid, tname in self.town_id_to_name.items():
            tgt_tokens = normalize_tokens(tname)
            k = len(tgt_tokens)
            for i in range(len(text_tokens) - k + 1):
                window = text_tokens[i:i + k]
                if fuzzy_ngram_match(window, tgt_tokens):
                    found_text_towns.append(tid)
                    break

        pincode_town = None
        if use_pincode and pincode and len(pincode) == 6:
            prefix = pincode[:2]
            if prefix == '96':
                pincode_town = 'T1'
            elif prefix == '97':
                pincode_town = 'T2'
            elif prefix == '98':
                pincode_town = 'T3'
            elif prefix == '99':
                pincode_town = 'OUT'

        conflict_flag = False

        if found_text_towns:
            text_town = found_text_towns[0]
            if pincode_town and pincode_town != 'OUT' and pincode_town != text_town:
                conflict_flag = True
                return text_town, 0.90, conflict_flag
            return text_town, 0.98, conflict_flag

        if pincode_town:
            return pincode_town, 0.95, False

        try:
            tfidf_pred = self.town_tfidf.predict([norm_raw])[0]
            return tfidf_pred, 0.70, False
        except Exception:
            pass

        return 'UNKNOWN', 0.30, False

    def resolve_locality(self, text: str, town_id: str, pincode_town: Optional[str] = None) -> Tuple[str, str, float]:
        """
        Locality resolution inside the resolved town (fuzzy n-gram match),
        then a low-confidence search in the pincode's town, then TF-IDF.
        Returns: (locality_id, locality_name, confidence)
        """
        if town_id in ['OUT', 'UNKNOWN']:
            return 'UNKNOWN', 'UNKNOWN', 0.0

        town_locs = [r for r in self.locality_records if r['town_id'] == town_id]
        text_tokens = normalize_tokens(text)

        for loc in town_locs:
            lid = loc['locality_id']
            lname = loc['locality_name']
            loc_tokens = normalize_tokens(lname)
            k = len(loc_tokens)
            for i in range(len(text_tokens) - k + 1):
                window = text_tokens[i:i + k]
                if fuzzy_ngram_match(window, loc_tokens):
                    return lid, lname, 0.98

        for loc in town_locs:
            lid = loc['locality_id']
            lname = loc['locality_name']
            loc_tokens = normalize_tokens(lname)
            if len(loc_tokens) >= 2:
                first_tok = loc_tokens[0]
                if len(first_tok) >= 4:
                    for tok in text_tokens:
                        if fuzzy_word_match(tok, first_tok):
                            return lid, lname, 0.88

        search_towns = [town_id]
        if pincode_town and pincode_town not in ['OUT', 'UNKNOWN', town_id]:
            search_towns.append(pincode_town)

        fallback_locs = [r for r in self.locality_records if r['town_id'] in search_towns]
        for loc in fallback_locs:
            lid = loc['locality_id']
            lname = loc['locality_name']
            loc_tokens = normalize_tokens(lname)
            k = len(loc_tokens)
            for i in range(len(text_tokens) - k + 1):
                window = text_tokens[i:i + k]
                if fuzzy_ngram_match(window, loc_tokens):
                    return lid, lname, 0.75
        # Step 4 TF-IDF locality fallback disabled to prevent assigning localities when none are mentioned
        return 'UNKNOWN', 'UNKNOWN', 0.0

    def resolve_landmarks_and_coordinates(
        self,
        norm_raw: str,
        latin_text: str,
        town_id: str,
        locality_id: str,
        pincode_town: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Landmark rule (v3):
          - Detect landmark types and relations in the text (unchanged).
          - For each detected type, take every POI of that type in the resolved town.
          - If the locality is known and has POIs of that type, keep only those
            (mode 'in_locality'). Otherwise keep all town POIs (mode 'candidates').
          - Every kept POI is returned with its own poi_x, poi_y.
          - POIs without coordinates go to 'dropped'.
        Returns: (landmarks, dropped)
        """
        if town_id in ['OUT', 'UNKNOWN']:
            return [], []

        found_lms: List[Dict[str, Any]] = []
        dropped_entities: List[Dict[str, Any]] = []
        found_spans: List[Tuple[int, int]] = []

        norm_text_tokens = normalize_tokens(norm_raw)
        latin_text_tokens = normalize_tokens(latin_text)

        # 1. Detect landmark types (detection logic unchanged from v2)
        detected_types = []
        for lm_type, syns in LANDMARK_TYPE_LEXICON.items():
            matched = False
            for syn in sorted(syns, key=len, reverse=True):
                if matched:
                    break

                for orig_text in [norm_raw, latin_text]:
                    if matched:
                        break
                    pat = r'\b' + re.escape(syn.lower()) + r'\b' if syn.isascii() else re.escape(syn)
                    m = re.search(pat, orig_text, re.IGNORECASE)
                    if m:
                        start, end = m.start(), m.end()
                        if lm_type == 'park':
                            orig_low = orig_text.lower()
                            if any(guard in orig_low for guard in LOCALITY_NAME_GUARDS):
                                continue
                        if any(max(start, s) < min(end, e) for (s, e) in found_spans):
                            continue
                        found_spans.append((start, end))
                        norm_rel, raw_rel, _ = self.labeller.find_relation(orig_text, start, end)
                        detected_types.append({
                            'landmark_type': lm_type,
                            'surface_span': orig_text[start:end],
                            'relation': norm_rel,
                            'raw_relation': raw_rel,
                        })
                        matched = True
                        break

                if matched:
                    break

                syn_tokens = normalize_tokens(syn)
                if not syn_tokens:
                    continue
                k = len(syn_tokens)

                for tok_list, orig_text in [(latin_text_tokens, latin_text), (norm_text_tokens, norm_raw)]:
                    if matched:
                        break
                    for i in range(len(tok_list) - k + 1):
                        window = tok_list[i:i + k]
                        if fuzzy_ngram_match(window, syn_tokens):
                            if lm_type == 'park':
                                orig_low = orig_text.lower()
                                if any(guard in orig_low for guard in LOCALITY_NAME_GUARDS):
                                    continue

                            pat = r'\b' + re.escape(syn.lower()) + r'\b'
                            m = re.search(pat, orig_text.lower())
                            if m:
                                start, end = m.start(), m.end()
                                if any(max(start, s) < min(end, e) for (s, e) in found_spans):
                                    continue
                                found_spans.append((start, end))
                                norm_rel, raw_rel, _ = self.labeller.find_relation(orig_text, start, end)
                            else:
                                norm_rel, raw_rel = 'unknown', None

                            detected_types.append({
                                'landmark_type': lm_type,
                                'surface_span': " ".join(window),
                                'relation': norm_rel,
                                'raw_relation': raw_rel,
                            })
                            matched = True
                            break

        # 2. Coordinate resolution (v3 rule)
        for item in detected_types:
            lm_type = item['landmark_type']
            surf = item['surface_span']
            rel = item['relation']

            # Step 1: every POI of this type in the resolved town
            town_pois = self.spatial_poi_df[
                (self.spatial_poi_df['town_id'] == town_id) &
                (self.spatial_poi_df['landmark_type'] == lm_type)
            ]

            # Pincode town fallback only if the resolved town has no POI of this type
            if len(town_pois) == 0 and pincode_town and pincode_town not in ['OUT', 'UNKNOWN', town_id]:
                town_pois = self.spatial_poi_df[
                    (self.spatial_poi_df['town_id'] == pincode_town) &
                    (self.spatial_poi_df['landmark_type'] == lm_type)
                ]

            if len(town_pois) == 0:
                dropped_entities.append({
                    'entity_name': surf,
                    'entity_type': lm_type,
                    'reason': f"No POIs of type '{lm_type}' in town '{town_id}'",
                })
                continue

            # Step 2: narrow to the locality if it is known and has such POIs
            mode = 'candidates'
            chosen = town_pois
            if locality_id != 'UNKNOWN':
                loc_pois = town_pois[town_pois['locality_id'] == locality_id]
                if len(loc_pois) > 0:
                    chosen = loc_pois
                    mode = 'in_locality'

            # Step 3: emit every chosen POI with its own coordinates
            n = len(chosen)
            for _, poi in chosen.iterrows():
                if pd.isna(poi['x']) or pd.isna(poi['y']):
                    dropped_entities.append({
                        'entity_name': str(poi['name']),
                        'entity_type': lm_type,
                        'reason': "Missing coordinates",
                    })
                    continue
                found_lms.append({
                    'landmark_name': str(poi['name']),
                    'landmark_type': lm_type,
                    'relation': rel,
                    'raw_relation': item.get('raw_relation'),
                    'surface_span': surf,
                    'poi_id': str(poi.get('poi_id', '')),
                    'poi_x': float(poi['x']),
                    'poi_y': float(poi['y']),
                    'x': float(poi['x']),
                    'y': float(poi['y']),
                    'resolution_mode': mode,
                    'candidate_count': n,
                })

        return found_lms, dropped_entities

    def predict(self, address_text: str, address_id: str = "UNKNOWN", use_pincode: bool = True) -> Dict[str, Any]:
        """Returns the standard output schema for one address."""
        latin_text, norm_raw, pincode = preprocess_address(address_text)

        valid_pincode = pincode if (pincode and len(pincode) == 6) else None
        pincode_town = None
        if valid_pincode:
            pincode_town = {'96': 'T1', '97': 'T2', '98': 'T3', '99': 'OUT'}.get(valid_pincode[:2])

        town_id, town_conf, conflict = self.resolve_town(latin_text, norm_raw, valid_pincode, use_pincode=use_pincode)
        town_name = self.town_id_to_name.get(town_id, town_id)

        loc_id, loc_name, loc_conf = self.resolve_locality(latin_text, town_id, pincode_town=pincode_town)

        loc_rec = next((r for r in self.locality_records if r['locality_id'] == loc_id), None)
        loc_x = float(loc_rec['centroid_x']) if loc_rec else None
        loc_y = float(loc_rec['centroid_y']) if loc_rec else None

        landmarks, dropped = self.resolve_landmarks_and_coordinates(
            norm_raw, latin_text, town_id, loc_id, pincode_town=pincode_town
        )

        return {
            'address_id': address_id,
            'raw_address': address_text,
            'town_id': town_id,
            'town_name': town_name,
            'locality_id': loc_id,
            'locality_name': loc_name,
            'locality_x': loc_x,
            'locality_y': loc_y,
            'landmarks': landmarks,
            'dropped': dropped,
            'confidences': {
                'town': town_conf,
                'locality': loc_conf,
                'landmarks': 0.90 if len(landmarks) > 0 else 0.0,
            },
            'conflict_flag': conflict,
        }