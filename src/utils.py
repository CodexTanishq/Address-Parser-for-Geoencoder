from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
import random
import numpy as np
import pandas as pd
import torch

from src import config


def set_seed(seed: int = config.RANDOM_SEED) -> None:
    """Ensure reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class AddressSpan:
    """Represents a recognized entity span within an address."""
    entity_type: str
    text: str
    start: int
    end: int
    confidence: str = config.CONF_HIGH
    source: str = "rule"
    match_score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def overlaps_with(self, other: "AddressSpan") -> bool:
        """Check whether two character spans overlap."""
        return max(self.start, other.start) < min(self.end, other.end)


def load_raw_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load addresses, towns, and localities as immutable raw data.
    Returns (addresses_df, towns_df, localities_df).
    """
    if not config.ADDRESSES_CSV.exists():
        raise FileNotFoundError(f"Addresses file not found: {config.ADDRESSES_CSV}")
    if not config.TOWNS_CSV.exists():
        raise FileNotFoundError(f"Towns file not found: {config.TOWNS_CSV}")
    if not config.LOCALITIES_CSV.exists():
        raise FileNotFoundError(f"Localities file not found: {config.LOCALITIES_CSV}")

    addresses_df = pd.read_csv(config.ADDRESSES_CSV)
    towns_df = pd.read_csv(config.TOWNS_CSV)
    localities_df = pd.read_csv(config.LOCALITIES_CSV)

    return addresses_df, towns_df, localities_df


def find_char_spans(text: str, subtext: str) -> List[Tuple[int, int]]:
    """
    Find all case-insensitive occurrences of subtext in text,
    returning list of (start, end) character offsets.
    """
    if not subtext or not text:
        return []
    spans = []
    text_lower = text.lower()
    sub_lower = subtext.lower()
    start = 0
    while True:
        idx = text_lower.find(sub_lower, start)
        if idx == -1:
            break
        spans.append((idx, idx + len(subtext)))
        start = idx + len(subtext)
    return spans
