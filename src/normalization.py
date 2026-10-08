import re
from typing import List
from src import config


def normalize_punctuation(text: str) -> str:
    """Standardize punctuation and unicode dashes without destroying boundaries."""
    # Standardize unicode dashes and hyphens
    text = re.sub(r"[\u2010\u2011\u2012\u2013\u2014\u2015\u2212]", "-", text)
    # Standardize quotes
    text = re.sub(r"[\u2018\u2019]", "'", text)
    text = re.sub(r"[\u201C\u201D]", '"', text)
    # Clean up duplicate punctuation (e.g., ,, or --)
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r"-\s*-+", "-", text)
    return text


def normalize_text(text: str) -> str:
    """
    Standardize text into a clean normalized form:
    - Lowercase
    - Punctuation normalization
    - Collapse redundant whitespace
    - Preserves numbers, slashes, hash signs, and meaningful words
    """
    if not text:
        return ""
    text = normalize_punctuation(text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()


def normalize_for_matching(text: str) -> str:
    """
    Normalize text specifically for fuzzy/gazetteer matching:
    - Lowercase
    - Replace punctuation with spaces
    - Expand known locality/street abbreviations
    - Collapse whitespace
    """
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    tokens = text.split()
    norm_tokens = [config.LOCALITY_ABBREV_MAP.get(t, t) for t in tokens]
    return " ".join(norm_tokens)
