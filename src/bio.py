import re
from typing import List, Tuple, Dict, Any
from src.utils import AddressSpan
from src import config

# Tokenizer matching words (alphanumerics, Indic scripts, internal hyphens/apostrophes) and punctuation
TOKEN_REGEX = re.compile(r"[\w\u0900-\u0D7F'-]+|[^\w\s]")


def tokenize_with_spans(text: str) -> List[Tuple[str, int, int]]:
    """
    Tokenize text into tokens with their (token_text, start_char, end_char) offsets.
    Punctuation marks (like commas, dashes, colons) are isolated as individual tokens.
    """
    tokens = []
    for match in TOKEN_REGEX.finditer(text):
        tokens.append((match.group(0), match.start(), match.end()))
    return tokens


def spans_to_bio(
    text: str,
    spans: List[AddressSpan],
    confidence_filter: str = config.CONF_HIGH,
) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Convert text and recognized AddressSpans into BIO token sequences.
    
    Args:
        text: Address text.
        spans: Recognized entity spans.
        confidence_filter: If set, only spans matching this confidence level
                           (or above) are labeled; others receive 'O'.
                           
    Returns:
        (tokens, bio_labels, token_metadata)
    """
    token_tuples = tokenize_with_spans(text)
    tokens: List[str] = [t[0] for t in token_tuples]
    bio_labels: List[str] = ["O"] * len(tokens)
    token_metadata: List[Dict[str, Any]] = []

    # Filter spans if confidence_filter is applied
    active_spans = [
        s for s in spans
        if confidence_filter is None or s.confidence == confidence_filter
    ]
    # Sort active spans by start offset
    active_spans.sort(key=lambda s: s.start)

    for i, (tok_str, tok_start, tok_end) in enumerate(token_tuples):
        matched_span: AddressSpan = None
        for s in active_spans:
            # Token falls within span
            if max(tok_start, s.start) < min(tok_end, s.end):
                matched_span = s
                break

        if matched_span:
            # Check if this is the first token inside the span
            # A token is the start (B-) if the previous token was not in the same span
            is_start = True
            if i > 0:
                prev_start, prev_end = token_tuples[i - 1][1], token_tuples[i - 1][2]
                if max(prev_start, matched_span.start) < min(prev_end, matched_span.end):
                    is_start = False

            prefix = "B-" if is_start else "I-"
            bio_labels[i] = f"{prefix}{matched_span.entity_type}"
            token_metadata.append({
                "token": tok_str,
                "label": bio_labels[i],
                "entity": matched_span.entity_type,
                "confidence": matched_span.confidence,
                "source": matched_span.source,
                "match_score": matched_span.match_score,
                "start": tok_start,
                "end": tok_end,
            })
        else:
            bio_labels[i] = "O"
            token_metadata.append({
                "token": tok_str,
                "label": "O",
                "entity": "O",
                "confidence": config.CONF_HIGH,
                "source": "unlabeled",
                "match_score": 1.0,
                "start": tok_start,
                "end": tok_end,
            })

    return tokens, bio_labels, token_metadata


def validate_bio_sequence(bio_labels: List[str]) -> bool:
    """
    Validate BIO constraints:
    - An 'I-X' label must be preceded by 'B-X' or 'I-X'.
    """
    for i, label in enumerate(bio_labels):
        if label.startswith("I-"):
            entity = label[2:]
            if i == 0:
                return False
            prev = bio_labels[i - 1]
            if prev not in (f"B-{entity}", f"I-{entity}"):
                return False
    return True
