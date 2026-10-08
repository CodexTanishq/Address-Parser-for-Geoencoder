import random
from typing import List, Dict, Any, Tuple, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from src import config
from src.utils import AddressSpan
from src.augmentation import AddressAugmenter


def load_canonical_spatial_database() -> Tuple[pd.DataFrame, pd.DataFrame, List[Dict[str, Any]]]:
    """
    Loads all canonical Localities and Landmarks/POIs with coordinates.
    Returns:
    - localities_df: with locality_id, locality_name, town_id, centroid_x, centroid_y
    - landmarks_df: with poi_id, town_id, landmark_type, name, x, y
    - all_canonical_entities: Unified list of entities for vector index indexing
    """
    localities_df = pd.read_csv(config.LOCALITIES_CSV)
    landmarks_df = pd.read_csv(config.LANDMARKS_POI_CSV)
    towns_df = pd.read_csv(config.TOWNS_CSV)

    town_map = dict(zip(towns_df["town_id"], towns_df["town_name"]))

    all_entities: List[Dict[str, Any]] = []

    # 1. Add Localities
    for _, r in localities_df.iterrows():
        lid = str(r["locality_id"])
        tid = str(r["town_id"])
        tname = town_map.get(tid, tid)
        lname = str(r["locality_name"])
        cx = float(r.get("centroid_x", 0.0))
        cy = float(r.get("centroid_y", 0.0))
        pin = str(r.get("pincode", ""))

        canonical_text = f"{lname}, {tname}"
        all_entities.append({
            "entity_id": lid,
            "entity_type": "LOCALITY",
            "name": lname,
            "town_id": tid,
            "town_name": tname,
            "pincode": pin,
            "x": cx,
            "y": cy,
            "canonical_text": canonical_text,
        })

    # 2. Add Landmarks/POIs
    for _, r in landmarks_df.iterrows():
        pid = str(r["poi_id"])
        tid = str(r["town_id"])
        tname = town_map.get(tid, tid)
        pname = str(r["name"])
        ptype = str(r["landmark_type"])
        px = float(r.get("x", 0.0))
        py = float(r.get("y", 0.0))

        canonical_text = f"{pname}, {ptype.replace('_', ' ').title()}, {tname}"
        all_entities.append({
            "entity_id": pid,
            "entity_type": "LANDMARK",
            "name": pname,
            "landmark_type": ptype,
            "town_id": tid,
            "town_name": tname,
            "pincode": "",
            "x": px,
            "y": py,
            "canonical_text": canonical_text,
        })

    return localities_df, landmarks_df, all_entities


def load_spatial_training_splits() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads training, validation, and test pairs strictly according to data/splits.csv (zero leakage).
    Constructs multi-task contrastive pairs:
    - Address -> Locality Target (canonical locality + town)
    - Address -> Landmark Target (canonical landmark + type + town)
    """
    splits_df = pd.read_csv(config.SPLITS_CSV)
    parsed_df = pd.read_csv(config.PARSED_ADDRESSES_CSV)
    spans_df = pd.read_csv(config.WEAK_SPANS_CSV)
    towns_df = pd.read_csv(config.TOWNS_CSV)
    town_map = dict(zip(towns_df["town_id"], towns_df["town_name"]))

    # Group spans by address_id for on-the-fly augmentation
    spans_by_addr: Dict[str, List[AddressSpan]] = {}
    for _, row in spans_df.iterrows():
        aid = str(row["address_id"])
        span = AddressSpan(
            entity_type=str(row["entity_type"]),
            text=str(row["entity_text"]),
            start=int(row["start"]),
            end=int(row["end"]),
            confidence=str(row.get("confidence", config.CONF_HIGH)),
            source=str(row.get("source", "rule")),
            match_score=float(row.get("match_score", 1.0)),
        )
        if aid not in spans_by_addr:
            spans_by_addr[aid] = []
        spans_by_addr[aid].append(span)

    merged = splits_df.merge(parsed_df, on="address_id", how="inner")

    train_pairs: List[Dict[str, Any]] = []
    val_pairs: List[Dict[str, Any]] = []
    test_pairs: List[Dict[str, Any]] = []

    for _, row in merged.iterrows():
        aid = str(row["address_id"])
        split = str(row["split"])
        raw_text = str(row["address_text"])
        tid = str(row.get("town_id", "T1"))
        tname = town_map.get(tid, str(row.get("town_name", tid)))
        lname = str(row.get("locality_name", ""))
        lms = str(row.get("landmarks", ""))

        spans = spans_by_addr.get(aid, [])

        # Target 1: Locality positive target
        if lname and lname != "nan" and lname != "UNKNOWN":
            loc_target = f"{lname}, {tname}"
            loc_pair = {
                "address_id": aid,
                "raw_address": raw_text,
                "target_type": "LOCALITY",
                "target_name": lname,
                "canonical_target": loc_target,
                "spans": spans,
            }
            if split == "train":
                train_pairs.append(loc_pair)
            elif split == "val":
                val_pairs.append(loc_pair)
            elif split == "test":
                test_pairs.append(loc_pair)

        # Target 2: Landmark positive target
        if lms and lms != "nan" and lms != "UNKNOWN":
            first_lm = lms.split("|")[0].strip() if "|" in lms else lms.strip()
            if first_lm:
                lm_target = f"{first_lm}, {tname}"
                lm_pair = {
                    "address_id": aid,
                    "raw_address": raw_text,
                    "target_type": "LANDMARK",
                    "target_name": first_lm,
                    "canonical_target": lm_target,
                    "spans": spans,
                }
                if split == "train":
                    train_pairs.append(lm_pair)
                elif split == "val":
                    val_pairs.append(lm_pair)
                elif split == "test":
                    test_pairs.append(lm_pair)

    # Assert zero data leakage
    train_aids = set(p["address_id"] for p in train_pairs)
    test_aids = set(p["address_id"] for p in test_pairs)
    assert len(train_aids.intersection(test_aids)) == 0, "Critical: Data leakage detected between train and test splits!"

    return train_pairs, val_pairs, test_pairs


class SpatialAddressPairDataset(Dataset):
    """
    Dataset yielding (Anchor, Positive) pairs with dynamic on-the-fly permutation
    and noise augmentation applied to the Anchor.
    """

    def __init__(
        self,
        pairs: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        augmenter: Optional[AddressAugmenter] = None,
        is_training: bool = False,
        max_length: int = 96,
    ):
        self.pairs = pairs
        self.tokenizer = tokenizer
        self.augmenter = augmenter
        self.is_training = is_training
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.pairs[idx]
        raw_text = item["raw_address"]
        spans = item["spans"]

        # Apply on-the-fly augmentation during training
        if self.is_training and self.augmenter is not None and spans:
            anchor_text, _, _ = self.augmenter.augment(raw_text, spans)
        else:
            anchor_text = raw_text

        return {
            "anchor_text": anchor_text,
            "positive_text": item["canonical_target"],
            "target_type": item["target_type"],
        }


def collate_spatial_pairs(batch: List[Dict[str, Any]], tokenizer: AutoTokenizer) -> Dict[str, torch.Tensor]:
    anchors = [b["anchor_text"] for b in batch]
    positives = [b["positive_text"] for b in batch]

    anchor_enc = tokenizer(
        anchors,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )

    pos_enc = tokenizer(
        positives,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    )

    return {
        "anchor_input_ids": anchor_enc["input_ids"],
        "anchor_attention_mask": anchor_enc["attention_mask"],
        "pos_input_ids": pos_enc["input_ids"],
        "pos_attention_mask": pos_enc["attention_mask"],
    }
