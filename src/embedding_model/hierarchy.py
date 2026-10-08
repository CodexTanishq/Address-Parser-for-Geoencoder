import random
from typing import Dict, List, Tuple, Any, Optional
import pandas as pd


class GazetteerIndex:
    """
    Manages the authoritative geographic hierarchy:
    TOWN -> LOCALITY -> [WEAK LANDMARKS / POIS]
    and supports hard negative candidate generation.
    """
    def __init__(self, towns_df: Optional[pd.DataFrame] = None, localities_df: Optional[pd.DataFrame] = None, weak_landmarks: Optional[Dict[str, List[str]]] = None):
        if towns_df is None or localities_df is None:
            from src.utils import load_raw_data
            _, default_towns, default_locs = load_raw_data()
            towns_df = towns_df if towns_df is not None else default_towns
            localities_df = localities_df if localities_df is not None else default_locs
        self.towns = {}
        for _, r in towns_df.iterrows():
            self.towns[str(r["town_id"])] = str(r["town_name"])

        # Locality mappings
        self.localities = {}  # loc_id -> (loc_name, town_id)
        self.town_to_localities = {}  # town_id -> list of loc_ids
        for _, r in localities_df.iterrows():
            lid = str(r["locality_id"])
            lname = str(r["locality_name"])
            tid = str(r["town_id"])
            self.localities[lid] = (lname, tid)
            if tid not in self.town_to_localities:
                self.town_to_localities[tid] = []
            self.town_to_localities[tid].append(lid)

        # Weak landmarks associated with locality (from parser outputs)
        if weak_landmarks is None:
            self.locality_to_landmarks = self._load_weak_landmarks_from_parsed()
        else:
            self.locality_to_landmarks = weak_landmarks

    @property
    def town_names(self) -> List[str]:
        return list(self.towns.values())

    def _load_weak_landmarks_from_parsed(self) -> Dict[str, List[str]]:
        import os
        from pathlib import Path
        lms_map = {}
        parsed_path = Path("outputs/parsed_addresses.csv")
        if parsed_path.exists():
            df = pd.read_csv(parsed_path)
            for _, r in df.iterrows():
                lname = str(r.get("locality_name", "")).strip()
                lm = str(r.get("landmarks", "")).strip()
                if lname and lname != "UNKNOWN" and lm and lm != "UNKNOWN" and lm != "nan":
                    if lname not in lms_map:
                        lms_map[lname] = set()
                    for item in lm.split("|"):
                        c = item.strip()
                        if c:
                            lms_map[lname].add(c)
        return {k: list(v) for k, v in lms_map.items()}

    def get_hard_negative_locality(self, correct_loc_id: str, correct_town_id: str) -> Tuple[str, str]:
        """
        Hard Negative Type 1: Same town, WRONG locality.
        Forces model to distinguish specific neighborhoods rather than generic municipal zone.
        """
        same_town_locs = [lid for lid in self.town_to_localities.get(correct_town_id, []) if lid != correct_loc_id]
        if same_town_locs:
            neg_id = random.choice(same_town_locs)
            neg_name, neg_tid = self.localities[neg_id]
            return neg_name, self.towns[neg_tid]
        
        # Fallback to any other locality
        other_locs = [lid for lid in self.localities if lid != correct_loc_id]
        neg_id = random.choice(other_locs)
        neg_name, neg_tid = self.localities[neg_id]
        return neg_name, self.towns[neg_tid]

    def get_hard_negative_town(self, correct_town_id: str) -> str:
        """Town negative: any other town in gazetteer."""
        other_towns = [tname for tid, tname in self.towns.items() if tid != correct_town_id]
        return random.choice(other_towns) if other_towns else "Unknown Town"

    def get_hard_negative_landmark(self, correct_landmark: str, loc_id: str) -> str:
        """Hard Negative Landmark: landmark from another locality or standard alternative."""
        all_landmarks = []
        for lid, l_list in self.locality_to_landmarks.items():
            if lid != loc_id:
                all_landmarks.extend(l_list)
        if all_landmarks:
            return random.choice(all_landmarks)
        return "Post Office"
