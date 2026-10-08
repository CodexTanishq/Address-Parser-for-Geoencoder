import re
from typing import List
from src.transliteration import transliterate_to_latin
from src.normalization import normalize_text


def preprocess_address(text: str, mode: str = "raw") -> str:
    """
    Experimental preprocessing modes for address text:
    A. 'raw': raw multilingual text intact
    B. 'transliterated': script transliteration into Latin baseline
    C. 'normalized_transliterated': transliterated + lowercased + normalized punctuation
    """
    if not text:
        return ""
    if mode == "raw":
        # Keep original characters, collapse excessive whitespace
        return re.sub(r"\s+", " ", text).strip()
    elif mode == "transliterated":
        return transliterate_to_latin(text).strip()
    elif mode == "normalized_transliterated":
        t = transliterate_to_latin(text)
        return normalize_text(t)
    elif mode == "without_pincode":
        # Experiment B: strip out 6-digit postal code to evaluate shortcut leakage
        stripped = re.sub(r"(?<!\d)\d{6}(?!\d)", "", text)
        return re.sub(r"\s+", " ", stripped).strip(" ,-")
    else:
        return text.strip()
