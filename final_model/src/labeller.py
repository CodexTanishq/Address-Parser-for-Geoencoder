"""
Milestone 1: Deterministic Labeller and Silver Label Generator
- Preprocessing: NFKC normalization, transliteration (Kannada/Devanagari to Latin),
  abbreviation expansion, pincode extraction.
- Lexicons: 14 landmark types with multilingual synonyms (Latin/Kannada/Hindi/Hinglish),
  relation word lexicon, town names, and 36 locality names.
- Output: Extracts town_id, locality_id, landmarks list, relations, and BIO token tags.
- Agreement report: Compares against addresses.csv gold town_id and pincode prefix.
"""
import sys
import io
import re
import unicodedata
import time
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd
import numpy as np



from final_model.src import config

# -------------------------------------------------------------
# 1. TRANSLITERATION TABLES (Devanagari & Kannada -> Latin)
# -------------------------------------------------------------
DEVANAGARI_MAP = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh',
    'ष': 'sh', 'स': 's', 'ह': 'h',
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', '्': '',
    'ं': 'n', 'ँ': 'n', 'ः': 'h',
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u',
    'ऊ': 'oo', 'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au'
}

KANNADA_MAP = {
    'ಕ': 'k', 'ಖ': 'kh', 'ಗ': 'g', 'ಘ': 'gh', 'ಙ': 'ng',
    'ಚ': 'ch', 'ಛ': 'chh', 'ಜ': 'j', 'ಝ': 'jh', 'ಞ': 'ny',
    'ಟ': 't', 'ಠ': 'th', 'ಡ': 'd', 'ಢ': 'dh', 'ಣ': 'n',
    'ತ': 't', 'ಥ': 'th', 'ದ': 'd', 'ಧ': 'dh', 'ನ': 'n',
    'ಪ': 'p', 'ಫ': 'ph', 'ಬ': 'b', 'ಭ': 'bh', 'ಮ': 'm',
    'ಯ': 'y', 'ರ': 'r', 'ಲ': 'l', 'ವ': 'v', 'ಶ': 'sh',
    'ಷ': 'sh', 'ಸ': 's', 'ಹ': 'h', 'ಳ': 'l',
    'ಾ': 'a', 'ಿ': 'i', 'ೀ': 'ee', 'ು': 'u', 'ೂ': 'oo',
    'ೆ': 'e', 'ೇ': 'e', 'ೈ': 'ai', 'ೊ': 'o', 'ೋ': 'o', 'ೌ': 'au', '್': '',
    'ಂ': 'n', 'ಃ': 'h',
    'ಅ': 'a', 'ಆ': 'aa', 'ಇ': 'i', 'ಈ': 'ee', 'ಉ': 'u',
    'ಊ': 'oo', 'ಎ': 'e', 'ಏ': 'e', 'ಐ': 'ai', 'ಒ': 'o', 'ಓ': 'o', 'ಔ': 'au'
}

def transliterate_text(text: str) -> str:
    """Transliterates Kannada and Devanagari characters to Latin for matching."""
    res = []
    for char in text:
        if char in DEVANAGARI_MAP:
            res.append(DEVANAGARI_MAP[char])
        elif char in KANNADA_MAP:
            res.append(KANNADA_MAP[char])
        else:
            res.append(char)
    return "".join(res)

# -------------------------------------------------------------
# 2. ABBREVIATIONS & CLEANING
# -------------------------------------------------------------
ABBREV_MAP = {
    r'\bngr\b': 'nagar',
    r'\bnagr\b': 'nagar',
    r'\bngar\b': 'nagar',
    r'\bcolny\b': 'colony',
    r'\bclny\b': 'colony',
    r'\bcol\b': 'colony',
    r'\blayt\b': 'layout',
    r'\blyt\b': 'layout',
    r'\blayou\b': 'layout',
    r'\bblk\b': 'block',
    r'\brd\b': 'road',
    r'\bmn\b': 'main',
    r'\bx\b': 'cross',
    r'\bcrs\b': 'cross',
    r'\bln\b': 'lane',
    r'\bst\b': 'street',
    r'\bbdvne\b': 'badavane',
    r'\bbadavne\b': 'badavane',
    r'\bshanti\b': 'shanthi',
    r'\bsiddeshwara\b': 'siddeshwara',
    r'\bsidddeshwara\b': 'siddeshwara',
    r'\bgov\b': 'govt',
    r'\bgovt\.\b': 'govt',
    r'\bopp\.\b': 'opp',
    r'\bnear\.\b': 'near',
}

def preprocess_address(raw_text: str) -> Tuple[str, str, Optional[str]]:
    """
    Normalizes address text, transliterates to Latin for matching,
    expands abbreviations, and extracts any 6-digit pincode.
    Returns: (clean_latin_text, raw_normalized_text, pincode)
    """
    norm_raw = unicodedata.normalize('NFKC', str(raw_text)).strip()
    
    # Extract 6-digit pincode if present
    pin_match = re.search(r'\b([1-9][0-9]{5})\b', norm_raw)
    pincode = pin_match.group(1) if pin_match else None
    
    # Matching text: transliterate native script to Latin and lowercase
    latin_text = transliterate_text(norm_raw).lower()
    
    for pat, repl in ABBREV_MAP.items():
        latin_text = re.sub(pat, repl, latin_text, flags=re.IGNORECASE)
        
    return latin_text, norm_raw, pincode

# -------------------------------------------------------------
# 3. RELATION WORDS LEXICON & NORMALIZATION
# -------------------------------------------------------------
# Normalizes to {near, opposite, behind, beside, in_front, next_to, unknown}
RELATION_PATTERNS = [
    (r'\b(opp|opposite|eduru|ke\s+saamne|ke\s+samne|के\s+सामने|ಎದುರು)\b', 'opposite'),
    (r'\b(bhd|behind|hinde|hindhe|ke\s+peeche|ke\s+piche|के\s+पीछे|ಹಿಂದೆ)\b', 'behind'),
    (r'\b(beside|ke\s+bagal|bagal|बगल|pakka)\b', 'beside'),
    (r'\b(next\s+to|adj|adjacent)\b', 'next_to'),
    (r'\b(in\s+front\s+of|in\s+front)\b', 'in_front'),
    (r'\b(nr|near|hattira|ke\s+pass|ke\s+paas|के\s+पास|ಹತ್ತಿರ)\b', 'near'),
]

RAW_RELATION_WORDS = [
    'opposite', 'opp', 'eduru', 'ke saamne', 'ke samne', 'के सामने', 'ಎದುರು',
    'behind', 'bhd', 'hinde', 'hindhe', 'ke peeche', 'ke piche', 'के पीछे', 'ಹಿಂದೆ',
    'beside', 'ke bagal', 'bagal', 'बगल',
    'next to', 'adj', 'adjacent', 'in front of', 'in front',
    'near', 'nr', 'hattira', 'ke pass', 'ke paas', 'के पास', 'ಹತ್ತಿರ'
]

# -------------------------------------------------------------
# 4. LANDMARK TYPE MULTILINGUAL SYNONYMS LEXICON
# -------------------------------------------------------------
LANDMARK_TYPE_LEXICON = {
    'ration_shop': [
        'ration shop', 'ration angadi', 'nyaya bele angadi', 'nyayabele angadi', 'nyayavile angadi',
        'ration dukan', 'raashan ki dukaan', 'ration ki dukaan', 'rashan ki dukaan', 'rashan dukan',
        'pds shop', 'pds sohp', 'pds sho,p', 'pds dukan', 'ration angdi', 'ಪಡಿತರ ಅಂಗಡಿ',
        'ನ್ಯಾಯಬೆಲೆ ಅಂಗಡಿ', 'ನ್ಯಾಯಬೆಲೆ', 'राशन की दुकान', 'राशन दुकान', 'पीडीएस दुकान'
    ],
    'bus_stop': [
        'bus stop', 'bus stand', 'bus stop hattira', 'bus stand hattira', 'bus nildana',
        'bus adda', 'bus addda', 'bas stand', 'bas stand hattira', 'ಬಸ್ ನಿಲ್ದಾಣ', 'ಬಸ್ ಸ್ಟಾಪ್',
        'ಬಸ್ ಸ್ಟ್ಯಾಂಡ್', 'बस स्टैंड', 'बस स्टॉप', 'बस अड्डा'
    ],
    'govt_school': [
        'govt school', 'government school', 'sarkari shaale', 'sarkari school',
        'sarakari prathamika shale', 'sarakari shale', 'sarkari prathmik vidyalaya',
        'sarkari vidyalaya', 'sarkari school ke pass', 'srkari shaale', 'sarkari shaaale',
        'government schaol', 'governemnt school', 'ಸರ್ಕಾರಿ ಶಾಲೆ', 'ಸರ್ಕಾರಿ ಪ್ರಾಥಮಿಕ ಶಾಲೆ',
        'सरकारी स्कूल', 'सरकारी विद्यालय', 'सरकारी पाठशाला'
    ],
    'masjid': [
        'masjid', 'mosque', 'masjeed', 'masidi', 'jami masjid', 'jama masjid', 'madina masjid',
        'noor masjid', 'मस्जिद', 'जामा मस्जिद', 'ಮಸೀದಿ'
    ],
    'hanuman_temple': [
        'hanuman temple', 'anjaneya temple', 'anjaneya swamy temple', 'maruti temple',
        'hanuman mandir', 'haanuman mandir', 'hnauman temple', 'hnuman temple',
        'hanuman temlpe', 'hanuman templle', 'hanumn mandir', 'anjaneya gudi',
        'anjaneya guid', 'anjjaneya gudi', 'maruti mandir', 'sankat mochan mandir',
        'anjaneya devalaya', 'ಆಂಜನೇಯ ಗುಡಿ', 'ಆಂಜನೇಯ ಸ್ವಾಮಿ ದೇವಾಲಯ', 'ಹನುಮಾನ್ ದೇವಸ್ಥಾನ',
        'हनुमान मंदिर', 'मारुति मंदिर', 'संकट मोचन'
    ],
    'medical_store': [
        'medical store', 'medicals', 'pharmacy', 'chemist', 'dawai ki dukaan', 'dawa ki dukan',
        'dawakhana', 'aushadhi kendra', 'aarogya kendra', 'औषधि केंद्र', 'दवाखाना', 'मेडिकल स्टोर',
        'दवाई की दुकान', 'दवा की दुकान', 'ಔಷಧಿ ಅಂಗಡಿ', 'ಮೆಡಿಕಲ್'
    ],
    'ganesha_temple': [
        'ganesh temple', 'ganesha temple', 'vinayaka temple', 'ganapathi temple',
        'ganapathi gudi', 'ganapathi guudi', 'gannapathi gudi', 'ganesha gudi',
        'vinayaka devalaya', 'ganesh mandir', 'vinayak mandir', 'gaanesh mandir',
        'ಗಣೇಶ ಗುಡಿ', 'ಗಣಪತಿ ಗುಡಿ', 'ವಿನಾಯಕ ದೇವಸ್ಥಾನ', 'गणेश मंदिर', 'विनायक मंदिर', 'गणपति मंदिर'
    ],
    'water_tank': [
        'water tank', 'tanki', 'pani ki tanki', 'paani ki taanki', 'pani tanki',
        'neerina totti', 'over head tank', 'overhead tank', 'overhead tannk', 'oht',
        'neerina tank', 'ನೀರಿನ ಟ್ಯಾಂಕ್', 'ನೀರಿನ ತೊಟ್ಟಿ', 'पानी की टंकी'
    ],
    'milk_dairy': [
        'milk dairy', 'dairy', 'doodh dairy', 'doodh kendra', 'milk booth', 'milk boooth',
        'nandini milk booth', 'nandini booth', 'nandini dairy', 'haalu kendra',
        'haalina dairy', 'haalina daiyr', 'haalina daory', 'ಹಾಲು ಒಕ್ಕೂಟ', 'ಹಾಲು ಡೈರಿ',
        'ಹಾಲು ಕೇಂದ್ರ', 'दूध डेयरी', 'दूध केंद्र', 'मिल्क बूथ'
    ],
    'community_hall': [
        'community hall', 'community hlal', 'samudaya bhavana', 'kalyana mantapa',
        'kalyna mantapa', 'choultry', 'baraat ghar', 'barat ghar', 'barat ghhar',
        'milan kendra', 'samudayik bhavan', 'ಸಮುದಾಯ ಭವನ', 'ಕಲ್ಯಾಣ ಮಂಟಪ', 'कल्याण मंडप',
        'सामुदायिक भवन', 'बारात घर', 'बारातघर'
    ],
    'park': [
        'children park', 'children purk', 'park', 'udhyanavana', 'udhyana', 'bagicha',
        'garden', 'baag', 'ಉದ್ಯಾನವನ', 'ಪಾರ್ಕ್', 'पार्क', 'बगीचा'
    ],
    'church': [
        'church', 'saint mary church', 'st mary church', 'st joseph church',
        'prarthanalaya', 'girja ghar', 'ಚರ್ಚ್', 'ಚರ್ಚ್', 'गिरिजाघर'
    ],
    'post_office': [
        'post office', 'posst office', 'psot office', 'post ofifce', 'dak ghar',
        'anche kacheri', 'tapaal kacheri', 'anche office', 'ಅಂಚೆ ಕಚೇರಿ', 'डाक घर',
        'पोस्ट ऑफिस', 'डाकघर'
    ],
    'petrol_bunk': [
        'petrol bunk', 'petrol pump', 'filling station', 'fuel station',
        'ಪೆಟ್ರೋಲ್ ಬಂಕ್', 'पेट्रोल पंप', 'पेट्रोल बंक'
    ]
}

# Negative guards: Avoid mistaking locality names for parks/gardens/lakes
LOCALITY_NAME_GUARDS = [
    'green park layout', 'royal gardens', 'palm meadows', 'lakeview phase 2',
    'lakeview', 'green park', 'sunrise enclave', 'orchid enclave'
]

# -------------------------------------------------------------
# 5. LABELLER IMPLEMENTATION
# -------------------------------------------------------------
class DeterministicLabeller:
    """
    Deterministic Silver Labeller using Gazetteer Fuzzy Matching, Multi-token N-grams,
    Multilingual Lexicons, and Pincode cues.
    """
    def __init__(self):
        self.towns_df = pd.read_csv(config.TOWNS_CSV)
        self.localities_df = pd.read_csv(config.LOCALITIES_CSV)
        self.landmarks_df = pd.read_csv(config.LANDMARKS_POI_CSV)
        
        # Build spatial nearest locality mapping
        from final_model.src.audit import build_spatial_poi_index
        self.spatial_poi_df = build_spatial_poi_index(self.localities_df, self.landmarks_df)
        
        # Mapping dictionaries
        self.town_id_to_name = dict(zip(self.towns_df['town_id'], self.towns_df['town_name']))
        self.locality_records = self.localities_df.to_dict(orient='records')
        
        # Unique locality names vs town mapping
        # Note: "Nehru Colony" is in both T1 and T2!
        self.loc_to_towns = {}
        for r in self.locality_records:
            lname = str(r['locality_name']).lower().strip()
            self.loc_to_towns.setdefault(lname, []).append(r['town_id'])
            
    def detect_pincode_town(self, pincode: Optional[str]) -> Optional[str]:
        if not pincode or len(pincode) != 6:
            return None
        prefix = pincode[:2]
        if prefix == '96':
            return 'T1'
        elif prefix == '97':
            return 'T2'
        elif prefix == '98':
            return 'T3'
        elif prefix == '99':
            return 'OUT'
        return None

    def is_out_address(self, text: str, pincode: Optional[str]) -> bool:
        if pincode and pincode.startswith('99'):
            return True
        low = text.lower()
        if low.startswith('village') or ' taluk ' in low or ' tehsil ' in low or ' district ' in low:
            return True
        return False

    def find_relation(self, text: str, span_start: int, span_end: int) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int, int]]]:
        """
        Finds relation word near a landmark span (looking 30 chars before and after).
        Returns: (normalized_relation, raw_relation_word, (char_start, char_end))
        """
        window_start = max(0, span_start - 35)
        window_end = min(len(text), span_end + 35)
        
        # Check window before span
        before_text = text[window_start:span_start]
        after_text = text[span_end:window_end]
        
        for pat, norm in RELATION_PATTERNS:
            m_before = list(re.finditer(pat, before_text, re.IGNORECASE))
            if m_before:
                last_m = m_before[-1]
                raw_w = last_m.group(0)
                actual_start = window_start + last_m.start()
                actual_end = window_start + last_m.end()
                return norm, raw_w, (actual_start, actual_end)
                
            m_after = list(re.finditer(pat, after_text, re.IGNORECASE))
            if m_after:
                first_m = m_after[0]
                raw_w = first_m.group(0)
                actual_start = span_end + first_m.start()
                actual_end = span_end + first_m.end()
                return norm, raw_w, (actual_start, actual_end)
                
        return 'unknown', None, None

    def extract_landmarks(self, norm_text: str, latin_text: str) -> List[Dict[str, Any]]:
        """
        Extracts landmark candidates with landmark_type, surface_span, relation, and span offsets.
        Guards against falsely capturing locality names (e.g. Green Park Layout).
        """
        landmarks = []
        found_spans = [] # track (start, end) to avoid overlapping landmark matches
        
        # Sort landmark keywords by length descending so longer phrases match first
        for lm_type, syns in LANDMARK_TYPE_LEXICON.items():
            for syn in sorted(syns, key=len, reverse=True):
                # Search both in latin_text and norm_text
                for target_txt in [latin_text, norm_text]:
                    pattern = r'\b' + re.escape(syn.lower()) + r'\b'
                    for match in re.finditer(pattern, target_txt.lower()):
                        start, end = match.start(), match.end()
                        surface_span = target_txt[start:end]
                        
                        # Guard: Avoid mistaking locality names for parks/gardens (e.g. Green Park Layout is not a park)
                        if lm_type == 'park':
                            context_window = target_txt[max(0, start - 15):min(len(target_txt), end + 15)].lower()
                            if any(guard in context_window for guard in LOCALITY_NAME_GUARDS):
                                continue

                            
                        # Overlap check
                        if any(max(start, s) < min(end, e) for (s, e) in found_spans):
                            continue
                            
                        found_spans.append((start, end))
                        
                        # Find attached relation word
                        norm_rel, raw_rel, rel_span = self.find_relation(target_txt, start, end)
                        
                        landmarks.append({
                            'landmark_type': lm_type,
                            'surface_span': surface_span,
                            'char_start': start,
                            'char_end': end,
                            'relation': norm_rel,
                            'raw_relation': raw_rel,
                            'relation_span': rel_span
                        })
                        break
        return landmarks

    def extract_locality(self, latin_text: str, candidate_town: Optional[str]) -> Tuple[Optional[str], Optional[str], Optional[Tuple[int, int]], float]:
        """
        Finds best locality match from 36 localities, prioritizing candidate_town.
        Returns: (locality_id, locality_name, (char_start, char_end), confidence)
        """
        best_lid = None
        best_lname = None
        best_span = None
        best_score = 0.0
        
        for loc in self.locality_records:
            lid = loc['locality_id']
            tid = loc['town_id']
            lname = loc['locality_name']
            
            # If candidate_town is known, only match localities in that town
            if candidate_town and candidate_town != 'OUT' and tid != candidate_town:
                continue
                
            norm_loc = lname.lower().strip()
            # Try exact word match
            pat = r'\b' + re.escape(norm_loc) + r'\b'
            m = re.search(pat, latin_text)
            if m:
                score = 1.0
                if score > best_score:
                    best_score = score
                    best_lid = lid
                    best_lname = lname
                    best_span = (m.start(), m.end())
            else:
                # Substring match if words > 1
                words = norm_loc.split()
                if len(words) >= 2:
                    partial = words[0]
                    if len(partial) >= 5 and re.search(r'\b' + re.escape(partial) + r'\b', latin_text):
                        score = 0.8
                        if score > best_score:
                            best_score = score
                            best_lid = lid
                            best_lname = lname
                            # approximate span
                            m_part = re.search(r'\b' + re.escape(partial) + r'\b', latin_text)
                            best_span = (m_part.start(), m_part.end())
                            
        return best_lid, best_lname, best_span, best_score

    def label_address(self, raw_address: str, gold_town_hint: Optional[str] = None) -> Dict[str, Any]:
        """
        Parses and labels an address deterministically.
        Generates structured predictions and BIO token tags.
        """
        latin_text, norm_raw, pincode = preprocess_address(raw_address)
        
        # 1. OUT Check
        if self.is_out_address(norm_raw, pincode):
            return {
                'town_id': 'OUT',
                'locality_id': 'UNKNOWN',
                'locality_name': 'UNKNOWN',
                'landmarks': [],
                'pincode': pincode,
                'confidence': {'town': 1.0, 'locality': 0.0, 'landmarks': 0.0},
                'clean_text': norm_raw,
                'tokens': norm_raw.split(),
                'bio_tags': ['O'] * len(norm_raw.split())
            }
            
        # 2. Town Resolution (Pincode prefix + Town Name mentions)
        town_id = self.detect_pincode_town(pincode)
        if not town_id:
            for tid, tname in self.town_id_to_name.items():
                if tname.lower() in latin_text:
                    town_id = tid
                    break
                    
        # 3. Locality Extraction
        loc_id, loc_name, loc_span, loc_conf = self.extract_locality(latin_text, candidate_town=town_id)
        
        # If town was unknown but locality is unique to one town, infer town
        if not town_id and loc_name:
            cands = self.loc_to_towns.get(loc_name.lower().strip(), [])
            if len(cands) == 1:
                town_id = cands[0]
                
        if not town_id:
            town_id = 'UNKNOWN'
            
        # 4. Landmarks Extraction
        raw_lms = self.extract_landmarks(norm_raw, latin_text)
        resolved_lms = []
        
        for lm in raw_lms:
            l_type = lm['landmark_type']
            surf = lm['surface_span']
            rel = lm['relation']
            raw_rel = lm['raw_relation']
            
            # Spatial Coordinate Resolution
            # Check exact key: (town_id, locality_id, landmark_type)
            poi_row = None
            res_mode = "unknown"
            
            if town_id != 'UNKNOWN' and loc_id and loc_id != 'UNKNOWN':
                exact_match = self.spatial_poi_df[
                    (self.spatial_poi_df['town_id'] == town_id) &
                    (self.spatial_poi_df['locality_id'] == loc_id) &
                    (self.spatial_poi_df['landmark_type'] == l_type)
                ]
                if len(exact_match) > 0:
                    poi_row = exact_match.iloc[0]
                    res_mode = "exact_key"
                    
            # Fallback: nearest_in_town
            if poi_row is None and town_id != 'UNKNOWN':
                town_pois = self.spatial_poi_df[
                    (self.spatial_poi_df['town_id'] == town_id) &
                    (self.spatial_poi_df['landmark_type'] == l_type)
                ]
                if len(town_pois) > 0:
                    if loc_id and loc_id != 'UNKNOWN':
                        # Find nearest to locality centroid
                        loc_rec = next((r for r in self.locality_records if r['locality_id'] == loc_id), None)
                        if loc_rec:
                            lx, ly = float(loc_rec['centroid_x']), float(loc_rec['centroid_y'])
                            town_pois = town_pois.copy()
                            town_pois['d'] = town_pois.apply(lambda r: (r['x'] - lx)**2 + (r['y'] - ly)**2, axis=1)
                            poi_row = town_pois.sort_values('d').iloc[0]
                            res_mode = "nearest_in_town"
                    else:
                        # Ambiguous: pick first candidate
                        poi_row = town_pois.iloc[0]
                        res_mode = "ambiguous"
                        
            resolved_lms.append({
                'landmark_type': l_type,
                'surface_span': surf,
                'relation': rel,
                'raw_relation': raw_rel,
                'poi_id': poi_row['poi_id'] if poi_row is not None else None,
                'poi_name': poi_row['name'] if poi_row is not None else None,
                'x': float(poi_row['x']) if poi_row is not None else None,
                'y': float(poi_row['y']) if poi_row is not None else None,
                'resolution_mode': res_mode,
                'char_start': lm['char_start'],
                'char_end': lm['char_end'],
                'relation_span': lm['relation_span']
            })
            
        # 5. BIO Tags Generation on Word Tokens
        tokens = norm_raw.split()
        bio_tags = ['O'] * len(tokens)
        
        # Helper to tag token spans
        # Reconstruct token character offsets in norm_raw
        char_idx = 0
        token_spans = []
        for t in tokens:
            pos = norm_raw.find(t, char_idx)
            token_spans.append((pos, pos + len(t)))
            char_idx = pos + len(t)
            
        def tag_range(c_start: int, c_end: int, tag_prefix: str):
            first = True
            for i, (t_s, t_e) in enumerate(token_spans):
                if max(t_s, c_start) < min(t_e, c_end):
                    bio_tags[i] = f"B-{tag_prefix}" if first else f"I-{tag_prefix}"
                    first = False

        if loc_span:
            tag_range(loc_span[0], loc_span[1], "LOCALITY")
            
        for lm in resolved_lms:
            tag_range(lm['char_start'], lm['char_end'], "LANDMARK")
            if lm['relation_span']:
                tag_range(lm['relation_span'][0], lm['relation_span'][1], "RELATION")
                
        # Tag Town if mentioned
        for tid, tname in self.town_id_to_name.items():
            if tname.lower() in norm_raw.lower():
                m_t = re.search(r'\b' + re.escape(tname) + r'\b', norm_raw, re.IGNORECASE)
                if m_t:
                    tag_range(m_t.start(), m_t.end(), "TOWN")

        return {
            'town_id': town_id,
            'locality_id': loc_id if loc_id else 'UNKNOWN',
            'locality_name': loc_name if loc_name else 'UNKNOWN',
            'landmarks': resolved_lms,
            'pincode': pincode,
            'confidence': {
                'town': 0.95 if town_id != 'UNKNOWN' else 0.3,
                'locality': loc_conf,
                'landmarks': 0.9 if len(resolved_lms) > 0 else 0.0
            },
            'clean_text': norm_raw,
            'tokens': tokens,
            'bio_tags': bio_tags
        }

def run_labeller_agreement_report():
    start_time = time.time()
    print("=" * 70)
    print("MILESTONE 1: DETERMINISTIC LABELLER & SILVER LABEL AGREEMENT REPORT")
    print("=" * 70)
    
    addresses_df = pd.read_csv(config.ADDRESSES_CSV)
    splits_df = pd.read_csv(config.SPLITS_CSV)
    merged = pd.merge(addresses_df, splits_df, on='account_id')
    merged['split'] = merged['split'].replace({'validation': 'val'})
    
    labeller = DeterministicLabeller()
    
    results = []
    print(f"Applying Deterministic Labeller to {len(merged)} addresses...")
    
    town_matches = 0
    town_total_known = 0
    out_matches = 0
    out_total = 0
    pincode_checked = 0
    pincode_agreements = 0
    
    for _, row in merged.iterrows():
        raw_text = str(row['address_text'])
        gold_tid = str(row['town_id'])
        
        pred = labeller.label_address(raw_text)
        pred_tid = pred['town_id']
        pred_pin = pred['pincode']
        
        # Check OUT agreement
        if gold_tid == 'OUT':
            out_total += 1
            if pred_tid == 'OUT':
                out_matches += 1
        elif gold_tid in ['T1', 'T2', 'T3']:
            town_total_known += 1
            if pred_tid == gold_tid:
                town_matches += 1
                
        # Check pincode agreement
        if pred_pin:
            pincode_checked += 1
            pin_tid = labeller.detect_pincode_town(pred_pin)
            if pin_tid == gold_tid:
                pincode_agreements += 1
                
        results.append({
            'address_id': row['address_id'],
            'account_id': row['account_id'],
            'split': row['split'],
            'address_text': raw_text,
            'gold_town_id': gold_tid,
            'silver_town_id': pred_tid,
            'silver_locality_id': pred['locality_id'],
            'silver_locality_name': pred['locality_name'],
            'silver_num_landmarks': len(pred['landmarks']),
            'silver_landmarks_json': pred['landmarks'],
            'tokens': pred['tokens'],
            'bio_tags': pred['bio_tags']
        })
        
    res_df = pd.DataFrame(results)
    
    # Save silver labels
    silver_out_path = config.OUTPUTS_DIR / "silver_labels.parquet"
    res_df.to_parquet(silver_out_path, index=False)
    print(f"[OK] Saved {len(res_df)} silver-labelled rows to {silver_out_path}")
    
    print("\n--- Silver Labeller Agreement with Gold Truth ---")
    print(f"In-Region Town Accuracy: {town_matches}/{town_total_known} ({town_matches/town_total_known*100:.2f}%)")
    print(f"OUT Village Detection Accuracy: {out_matches}/{out_total} ({out_matches/out_total*100:.2f}%)")
    print(f"Pincode Prefix vs Gold Town Agreement: {pincode_agreements}/{pincode_checked} ({pincode_agreements/pincode_checked*100:.2f}%)")
    
    # Locality and Landmark statistics
    loc_resolved = (res_df['silver_locality_id'] != 'UNKNOWN').sum()
    lm_resolved = (res_df['silver_num_landmarks'] > 0).sum()
    print(f"Addresses with Locality Resolved: {loc_resolved}/{len(res_df)} ({loc_resolved/len(res_df)*100:.2f}%)")
    print(f"Addresses with >=1 Landmark Resolved: {lm_resolved}/{len(res_df)} ({lm_resolved/len(res_df)*100:.2f}%)")
    
    elapsed = time.time() - start_time
    print(f"\n[Milestone 1 Elapsed Time: {elapsed:.2f}s]")
    return res_df

if __name__ == "__main__":
    run_labeller_agreement_report()
