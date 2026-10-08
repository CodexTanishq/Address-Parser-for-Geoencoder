import re
from typing import List, Tuple, Dict
from indic_transliteration import sanscript

SCRIPT_RANGES = {
    "Devanagari": (0x0900, 0x097F, sanscript.DEVANAGARI),
    "Bengali": (0x0980, 0x09FF, sanscript.BENGALI),
    "Gurmukhi": (0x0A00, 0x0A7F, sanscript.GURMUKHI),
    "Gujarati": (0x0A80, 0x0AFF, sanscript.GUJARATI),
    "Oriya": (0x0B00, 0x0B7F, sanscript.ORIYA),
    "Tamil": (0x0B80, 0x0BFF, sanscript.TAMIL),
    "Telugu": (0x0C00, 0x0C7F, sanscript.TELUGU),
    "Kannada": (0x0C80, 0x0CFF, sanscript.KANNADA),
    "Malayalam": (0x0D00, 0x0D7F, sanscript.MALAYALAM),
}

# Regex matching any character from Indic Unicode blocks
INDIC_PATTERN = re.compile(r"([\u0900-\u0D7F]+)")


def detect_scripts(text: str) -> List[str]:
    """Detect all scripts present in text."""
    scripts = set()
    for ch in text:
        cp = ord(ch)
        for sname, (start_cp, end_cp, _) in SCRIPT_RANGES.items():
            if start_cp <= cp <= end_cp:
                scripts.add(sname)
                break
        if "a" <= ch.lower() <= "z":
            scripts.add("Latin")
    return sorted(list(scripts))


def transliterate_indic_chunk(chunk: str) -> str:
    """Transliterate a contiguous chunk of Indic text to Latin using ITRANS scheme."""
    res = chunk
    for sname, (_, _, script_code) in SCRIPT_RANGES.items():
        try:
            res = sanscript.transliterate(res, script_code, sanscript.ITRANS)
        except Exception:
            pass
    # Clean up any residual ITRANS diacritics or special vowel markings to simple Latin
    # e.g., è -> e, ò -> o, .D -> D, etc.
    res = res.replace("è", "e").replace("ò", "o").replace("M", "m")
    return res


def transliterate_to_latin(text: str) -> str:
    """
    Transliterate all Indic characters in text to Latin while preserving English,
    punctuation, and numbers completely intact.
    """
    if not any(0x0900 <= ord(c) <= 0x0D7F for c in text):
        return text

    def _replace_match(m):
        return transliterate_indic_chunk(m.group(1))

    return INDIC_PATTERN.sub(_replace_match, text)


def transliterate_with_mapping(
    text: str,
) -> Tuple[str, List[Tuple[Tuple[int, int], Tuple[int, int]]]]:
    """
    Transliterate text to Latin and return both the transliterated text
    and a list of span alignments: ((orig_start, orig_end), (trans_start, trans_end)).
    This guarantees exact span recovery back to the original text.
    """
    if not any(0x0900 <= ord(c) <= 0x0D7F for c in text):
        return text, []

    mappings = []
    last_idx = 0
    trans_text = ""

    for match in INDIC_PATTERN.finditer(text):
        start, end = match.span()
        # Append preceding non-indic characters
        prefix = text[last_idx:start]
        trans_text += prefix

        indic_word = match.group(1)
        t_word = transliterate_indic_chunk(indic_word)

        t_start = len(trans_text)
        trans_text += t_word
        t_end = len(trans_text)

        mappings.append(((start, end), (t_start, t_end)))
        last_idx = end

    trans_text += text[last_idx:]
    return trans_text, mappings


def map_trans_span_to_orig(
    trans_start: int,
    trans_end: int,
    mappings: List[Tuple[Tuple[int, int], Tuple[int, int]]],
    orig_len: int,
    trans_len: int,
) -> Tuple[int, int]:
    """
    Given a span [trans_start, trans_end] in transliterated text,
    calculate the corresponding [orig_start, orig_end] in original text.
    """
    if not mappings:
        return trans_start, trans_end

    def _map_point(tp: int) -> int:
        for (os, oe), (ts, te) in mappings:
            if ts <= tp <= te:
                # Interpolate inside the chunk
                if te == ts:
                    return os
                ratio = (tp - ts) / (te - ts)
                return int(round(os + ratio * (oe - os)))
            elif tp < ts:
                # Point is in the non-indic text before this mapping
                # Compute distance before ts
                delta = ts - tp
                return os - delta
        # After the last mapping
        last_orig = mappings[-1][0][1]
        last_trans = mappings[-1][1][1]
        return last_orig + (tp - last_trans)

    orig_start = max(0, min(orig_len, _map_point(trans_start)))
    orig_end = max(orig_start, min(orig_len, _map_point(trans_end)))
    return orig_start, orig_end
