"""Build train BIO labels from gated AI labels and fixed fallback labels."""

from pathlib import Path
import re

import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
TRAIN = PROJECT / "data" / "labels" / "train_bio.csv"
HOLDOUT = PROJECT / "temp_code" / "holdout_ids.csv"
AI_ROOT = PROJECT / "temp_code" / "labeller_out"
OUTPUT = PROJECT / "data" / "labels" / "train_bio_merged.csv"
VALID_TAG = re.compile(r"^(O|[BI]-(TOWN|LANDMARK|LOCALITY|RELATION|PINCODE))$")


def fix_orphan_i(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("word_index").copy()
    previous = "O"
    for index, tag in group["tag"].items():
        if tag.startswith("I-") and previous not in {f"B-{tag[2:]}", f"I-{tag[2:]}"}:
            group.at[index, "tag"] = f"B-{tag[2:]}"
        previous = group.at[index, "tag"]
    return group


def no_invalid_bio(group: pd.DataFrame) -> bool:
    previous = "O"
    for tag in group.sort_values("word_index")["tag"]:
        if not VALID_TAG.fullmatch(tag):
            return False
        if tag.startswith("I-") and previous not in {f"B-{tag[2:]}", f"I-{tag[2:]}"}:
            return False
        previous = tag
    return True


def main() -> None:
    fallback = pd.read_csv(TRAIN, dtype=str).fillna("")
    holdout = set(pd.read_csv(HOLDOUT, dtype=str)["address_id"])
    required = {"address_id", "word_index", "word", "tag"}
    if not required.issubset(fallback.columns):
        raise ValueError(f"{TRAIN.name} is missing required columns")
    fallback = fallback[~fallback["address_id"].isin(holdout)].copy()
    fallback_key = fallback.set_index(["address_id", "word_index"])
    ai_tags: dict[tuple[str, str], str] = {}

    for number in range(1, 15):
        accepted_path = AI_ROOT / f"check2_{number:03d}" / "accepted_ai_labels.csv"
        rejected_path = AI_ROOT / f"check2_{number:03d}" / "rejected_ai_labels.csv"
        for path in (accepted_path, rejected_path):
            table = pd.read_csv(path, dtype=str).fillna("")
            if not required.issubset(table.columns):
                raise ValueError(f"{path} is missing required columns")
        accepted = pd.read_csv(accepted_path, dtype=str).fillna("")
        for row in accepted.itertuples(index=False):
            key = (row.address_id, row.word_index)
            if key not in fallback_key.index or fallback_key.at[key, "word"] != row.word:
                raise ValueError(f"Accepted AI words do not match fixed labels: {key}")
            ai_tags[key] = row.tag

    merged = fallback.copy()
    merged["tag"] = [ai_tags.get((row.address_id, row.word_index), row.tag) for row in merged.itertuples(index=False)]
    merged = pd.concat([fix_orphan_i(group) for _, group in merged.groupby("address_id", sort=False)], ignore_index=True)
    merged = merged.sort_values(["address_id", "word_index"], key=lambda col: col.astype(int) if col.name == "word_index" else col).reset_index(drop=True)

    address_count = merged["address_id"].nunique()
    duplicate_words = merged.duplicated(["address_id", "word_index"]).any()
    holdout_present = merged["address_id"].isin(holdout).any()
    reference_words = fallback[["address_id", "word_index", "word"]].sort_values(["address_id", "word_index"])
    merged_words = merged[["address_id", "word_index", "word"]].sort_values(["address_id", "word_index"])
    words_identical = reference_words.reset_index(drop=True).equals(merged_words.reset_index(drop=True))
    bio_valid = all(no_invalid_bio(group) for _, group in merged.groupby("address_id", sort=False))
    print(f"total addresses: {address_count}")
    print(f"duplicate address_id/word_index: {duplicate_words}")
    print(f"hold-out address inside: {holdout_present}")
    print(f"words identical to train_bio.csv: {words_identical}")
    print(f"invalid BIO transition left: {not bio_valid}")
    if address_count != 2075 or duplicate_words or holdout_present or not words_identical or not bio_valid:
        raise ValueError("Merged-label checks failed; output was not written")
    OUTPUT.parent.mkdir(exist_ok=True)
    merged.to_csv(OUTPUT, index=False)
    print(f"wrote: {OUTPUT}")


if __name__ == "__main__":
    main()
