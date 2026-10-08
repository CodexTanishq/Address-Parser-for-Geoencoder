import random
import re
from typing import List, Tuple, Dict, Any, Optional
from src.utils import AddressSpan
from src.bio import spans_to_bio
from src import config


class AddressChunk:
    """Represents a component chunk of an address (entity or non-entity delimiter)."""
    def __init__(self, text: str, span: Optional[AddressSpan] = None):
        self.text = text
        self.span = span

    @property
    def is_entity(self) -> bool:
        return self.span is not None


class AddressAugmenter:
    """
    Augmentation engine for Indian addresses to prevent positional overfitting:
    1. Randomly shuffles address fields (e.g., placing Pincode, Locality before House Number).
    2. Injects realistic noise (omitted/swapped punctuation, typos, case variations).
    3. Dynamically recalculates character span offsets so entity annotations
       (HOUSE_NUMBER, STREET, LOCALITY, WARD, PINCODE, TOWN, LANDMARK) stay 100% accurate.
    """

    TYPO_SWAPS = {
        "a": ["e", "s", "q"],
        "e": ["a", "r", "w"],
        "i": ["o", "u", "k"],
        "o": ["i", "p", "l"],
        "u": ["y", "i", "j"],
        "n": ["m", "b", "h"],
        "m": ["n", "j", "k"],
    }

    DELIMITERS = [", ", " - ", " / ", " ", ",  ", " "]

    def __init__(
        self,
        augmentation_prob: float = config.AUGMENTATION_PROB,
        shuffle_prob: float = config.SHUFFLE_PROB,
        noise_prob: float = config.NOISE_PROB,
    ):
        self.augmentation_prob = augmentation_prob
        self.shuffle_prob = shuffle_prob
        self.noise_prob = noise_prob

    def decompose_address(
        self, text: str, spans: List[AddressSpan]
    ) -> List[AddressChunk]:
        """
        Decomposes full address text into contiguous chunks of entities and interstitial delimiters.
        """
        # Sort spans by start offset
        valid_spans = [s for s in spans if s.start < s.end and s.end <= len(text)]
        valid_spans.sort(key=lambda s: s.start)

        # Merge or filter overlapping spans to keep clean boundaries
        non_overlapping: List[AddressSpan] = []
        last_end = 0
        for s in valid_spans:
            if s.start >= last_end:
                non_overlapping.append(s)
                last_end = s.end

        chunks: List[AddressChunk] = []
        curr = 0
        for s in non_overlapping:
            if s.start > curr:
                delim_text = text[curr:s.start]
                chunks.append(AddressChunk(text=delim_text, span=None))
            chunks.append(AddressChunk(text=text[s.start:s.end], span=s))
            curr = s.end

        if curr < len(text):
            chunks.append(AddressChunk(text=text[curr:], span=None))

        return chunks

    def apply_character_noise(self, text: str) -> str:
        """Injects realistic typing errors and casing jitter."""
        if not text or len(text) < 4:
            return text

        chars = list(text)
        action = random.random()

        if action < 0.35:
            # Random single-char substitution
            idx = random.randint(0, len(chars) - 1)
            char_lower = chars[idx].lower()
            if char_lower in self.TYPO_SWAPS:
                sub = random.choice(self.TYPO_SWAPS[char_lower])
                chars[idx] = sub.upper() if chars[idx].isupper() else sub
        elif action < 0.60:
            # Case variation (lowercase or uppercase a word)
            if random.random() < 0.5:
                return text.lower()
            else:
                return text.title()
        elif action < 0.85 and len(chars) > 5:
            # Omit one non-essential character
            idx = random.randint(1, len(chars) - 2)
            if not chars[idx].isdigit():  # Do not drop pincode / house digits
                chars.pop(idx)

        return "".join(chars)

    def augment(
        self,
        raw_text: str,
        spans: List[AddressSpan],
    ) -> Tuple[str, List[str], List[str]]:
        """
        Performs end-to-end address permutation and noise injection,
        returning (augmented_text, tokens, bio_labels).
        """
        if random.random() > self.augmentation_prob:
            # Return unaugmented BIO tokens
            tokens, bio_labels, _ = spans_to_bio(raw_text, spans)
            return raw_text, tokens, bio_labels

        chunks = self.decompose_address(raw_text, spans)
        entity_chunks = [c for c in chunks if c.is_entity]

        if len(entity_chunks) < 2:
            tokens, bio_labels, _ = spans_to_bio(raw_text, spans)
            return raw_text, tokens, bio_labels

        # 1. Permute / Shuffle entity chunks
        if random.random() < self.shuffle_prob:
            # Check if we shuffle completely or just swap sections
            strategy = random.choice(["reverse", "shuffle", "move_pincode_first", "locality_first"])
            if strategy == "reverse":
                entity_chunks.reverse()
            elif strategy == "shuffle":
                random.shuffle(entity_chunks)
            elif strategy == "move_pincode_first":
                # Find pincode or town and bring to front
                pin_or_town = [c for c in entity_chunks if c.span.entity_type in ("PINCODE", "TOWN")]
                others = [c for c in entity_chunks if c.span.entity_type not in ("PINCODE", "TOWN")]
                random.shuffle(pin_or_town)
                entity_chunks = pin_or_town + others
            elif strategy == "locality_first":
                # Find locality or ward and bring to front
                loc_or_ward = [c for c in entity_chunks if c.span.entity_type in ("LOCALITY", "WARD")]
                others = [c for c in entity_chunks if c.span.entity_type not in ("LOCALITY", "WARD")]
                entity_chunks = loc_or_ward + others

        # 2. Reconstruct address with dynamic delimiter choice and noise
        reconstructed_parts: List[str] = []
        new_spans: List[AddressSpan] = []
        current_offset = 0

        for i, c in enumerate(entity_chunks):
            chunk_text = c.text

            # Apply character noise to non-pincode, non-house-number entities with probability
            if random.random() < self.noise_prob:
                if c.span.entity_type not in ("PINCODE", "HOUSE_NUMBER"):
                    chunk_text = self.apply_character_noise(chunk_text)

            start = current_offset
            end = start + len(chunk_text)
            new_span = AddressSpan(
                entity_type=c.span.entity_type,
                text=chunk_text,
                start=start,
                end=end,
                confidence=c.span.confidence,
                source=c.span.source,
                match_score=c.span.match_score,
            )
            new_spans.append(new_span)
            reconstructed_parts.append(chunk_text)
            current_offset = end

            # Add separator between entities
            if i < len(entity_chunks) - 1:
                delim = random.choice(self.DELIMITERS)
                reconstructed_parts.append(delim)
                current_offset += len(delim)

        augmented_text = "".join(reconstructed_parts)

        # 3. Obtain 100% aligned BIO token labels from new spans
        tokens, bio_labels, _ = spans_to_bio(augmented_text, new_spans)

        return augmented_text, tokens, bio_labels
