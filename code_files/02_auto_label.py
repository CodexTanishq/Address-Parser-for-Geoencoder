"""Create conservative BIO labels for train and validation addresses only."""

from pathlib import Path
import random
import re

import pandas as pd
from rapidfuzz import fuzz, process


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
LABEL_DIR = DATA_DIR / "labels"
INPUT_FILES = {"train": DATA_DIR / "cleaned" / "train_cleaned.csv", "validation": DATA_DIR / "cleaned" / "validation_cleaned.csv"}
TOKEN = re.compile(r"\d+(?:st|nd|rd|th)\b|\d+|[a-z]+")
SHORT_FORMS = {"ngr": "nagar", "nagr": "nagar", "layt": "layout", "lyt": "layout", "col": "colony", "colny": "colony"}

# These requested type mappings are assumptions, not POI-name matches.
ASSUMPTION_TYPE_MAP = {
    "landmark:mosque": "masjid", "landmark:school": "govt_school",
    "landmark:dairy": "milk_dairy", "landmark:petrol_pump": "petrol_bunk",
    "landmark:wedding_hall": "community_hall",
}
GROUP_TYPE_MAP = {
    "landmark:bus_stop": "bus_stop", "landmark:church": "church", "landmark:medical_store": "medical_store",
    "landmark:park": "park", "landmark:post_office": "post_office", "landmark:ration_shop": "ration_shop",
    "landmark:water_tank": "water_tank", **ASSUMPTION_TYPE_MAP,
}
RELATION_WORDS = {
    "relation:near": {"near", "nr", "hattira", "samipa", "bali", "nazdik", "paas", "pasa"},
    "relation:opposite": {"opposite", "opp", "samane", "eduru"},
    "relation:behind": {"behind", "piche", "himde"},
    "relation:beside": {"beside", "bagala", "pakka"},
    "relation:next_to": {"next", "nextto"},
}
LANDMARK_WORDS = {
    "landmark:temple": {"temple", "mandir", "mamdira", "devasthana", "ganesh", "ganesha", "ganapati", "vinayaka", "hanuman", "hanumana", "anjaneya"},
    "landmark:mosque": {"mosque", "masjid", "masjida", "masidi"},
    "landmark:church": {"church", "charch", "charcha"},
    "landmark:school": {"school", "shaale", "shale", "skula"},
    "landmark:bus_stop": {"bus", "bas", "basa", "stop", "stopa", "busstop", "nildana"},
    "landmark:park": {"park", "parka", "udyanavana"},
    "landmark:post_office": {"post", "office", "postoffice", "dakaghara", "amche"},
    "landmark:medical_store": {"medical", "medicine", "pharmacy", "medikala", "medikal", "stora"},
    "landmark:ration_shop": {"ration", "rashana", "nyayabele"},
    "landmark:dairy": {"dairy", "milk", "deyari", "dairi", "dudha", "halina"},
    "landmark:water_tank": {"water", "tank", "watertank", "tamki", "tyamk"},
    "landmark:petrol_pump": {"petrol", "pump", "petrolpump", "fuel", "bunk", "petrola", "bamk"},
    "landmark:wedding_hall": {"mantapa", "mamtapa", "barata", "ghara"},
}


def words(text: str) -> list[str]:
    return TOKEN.findall(str(text).lower())


def expanded(text: str) -> str:
    return " ".join(SHORT_FORMS.get(word, word) for word in words(text))


def apply_bio(tags: list[str], indices: list[int], label: str) -> None:
    for position, index in enumerate(indices):
        tags[index] = f"{'B' if position == 0 else 'I'}-{label}"


def phrase_positions(tokens: list[str], phrase: str) -> list[int] | None:
    target = words(expanded(phrase))
    for start in range(len(tokens) - len(target) + 1):
        if tokens[start:start + len(target)] == target:
            return list(range(start, start + len(target)))
    return None


def town_positions(tokens: list[str], town_name: str) -> list[int] | None:
    exact = phrase_positions(tokens, town_name)
    if exact:
        return exact
    target = expanded(town_name)
    for start in range(len(tokens)):
        for length in range(1, min(4, len(tokens) - start) + 1):
            if fuzz.ratio(" ".join(tokens[start:start + length]), target) >= 86:
                return list(range(start, start + length))
    return None


def locality_match(tokens: list[str], choices: list[tuple[str, str]]) -> tuple[list[int] | None, str]:
    """Return exact, fuzzy, conflict, or no_locality without cross-town guesses."""
    exact_hits = [(locality_id, name, phrase_positions(tokens, name)) for locality_id, name in choices]
    exact_hits = [hit for hit in exact_hits if hit[2]]
    if len(exact_hits) == 1:
        return exact_hits[0][2], "sure"
    if len(exact_hits) > 1:
        return None, "conflict"

    address = " ".join(tokens)
    scored = []
    for locality_id, name in choices:
        score = fuzz.partial_ratio(expanded(name), address)
        if score >= 88:
            scored.append((score, locality_id, name))
    scored.sort(reverse=True)
    if not scored:
        return None, "no_locality"
    if len(scored) > 1 and scored[0][0] - scored[1][0] <= 2:
        return None, "conflict"
    positions = phrase_positions(tokens, scored[0][2])
    if positions:
        return positions, "fuzzy"
    return None, "no_locality"


def landmark_type(group: str, tokens: list[str], indices: list[int]) -> str | None:
    if group != "landmark:temple":
        return GROUP_TYPE_MAP.get(group)
    span_words = {tokens[index] for index in indices}
    if span_words & {"ganesh", "ganesha", "ganapati", "vinayaka"}:
        return "ganesha_temple"
    if span_words & {"hanuman", "hanumana", "anjaneya"}:
        return "hanuman_temple"
    return "temple_unknown"


def label_address(row: pd.Series, town_names: dict[str, str], locality_choices: dict[str, list[tuple[str, str]]], valid_pincodes: set[str]) -> tuple[list[dict], dict]:
    tokens = words(row.address_text)
    match_tokens = [SHORT_FORMS.get(token, token) for token in tokens]
    tags = ["O"] * len(tokens)
    types = [""] * len(tokens)
    quality = "no_locality"
    town_id = str(row.town_id)
    groups = set(filter(None, str(row.matched_groups).split(";")))

    for index, token in enumerate(tokens):
        if re.fullmatch(r"\d{5,6}", token):
            tags[index] = "B-PINCODE"

    if town_id != "OUT":
        town_span = town_positions(match_tokens, town_names[town_id])
        if town_span:
            apply_bio(tags, town_span, "TOWN")
        locality_span, quality = locality_match(match_tokens, locality_choices[town_id])
        if locality_span:
            apply_bio(tags, locality_span, "LOCALITY")
    locality_indices = {index for index, tag in enumerate(tags) if tag.endswith("LOCALITY")}

    relation_indices = []
    for group, spellings in RELATION_WORDS.items():
        if group in groups:
            relation_indices.extend(index for index, token in enumerate(tokens) if token in spellings)
    for index in relation_indices:
        if index not in locality_indices:
            tags[index] = "B-RELATION"

    for group, spellings in LANDMARK_WORDS.items():
        if group not in groups:
            continue
        candidates = [index for index, token in enumerate(tokens) if token in spellings and index not in locality_indices]
        active = [index for index in candidates if any(abs(index - relation) <= 3 for relation in relation_indices)]
        if not active:
            continue
        landmark_kind = landmark_type(group, tokens, active)
        apply_bio(tags, active, "LANDMARK")
        for index in active:
            types[index] = landmark_kind

    records = []
    for index, (token, tag, kind) in enumerate(zip(tokens, tags, types)):
        records.append({
            "address_id": row.address_id, "town_id": town_id, "word_index": index, "word": token, "tag": tag,
            "landmark_type": kind, "pincode_is_real": bool(tag == "B-PINCODE" and token in valid_pincodes),
            "label_quality": quality,
        })
    review = {"address_id": row.address_id, "town_id": town_id, "label_quality": quality,
              "labelled_address": " ".join(f"{token}/{tag}" for token, tag in zip(tokens, tags))}
    return records, review


def process_table(table: pd.DataFrame, town_names: dict[str, str], locality_choices: dict[str, list[tuple[str, str]]], valid_pincodes: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    labels, reviews = [], []
    for _, row in table.iterrows():
        address_labels, review = label_address(row, town_names, locality_choices, valid_pincodes)
        labels.extend(address_labels)
        reviews.append(review)
    return pd.DataFrame(labels), pd.DataFrame(reviews)


def main() -> None:
    towns = pd.read_csv(DATA_DIR / "towns.csv", dtype=str)
    localities = pd.read_csv(DATA_DIR / "localities.csv", dtype=str)
    pois = pd.read_csv(DATA_DIR / "landmarks_poi.csv", dtype=str)
    required_types = set(pois["landmark_type"])
    if len(required_types) != 14:
        raise ValueError("landmarks_poi.csv does not contain the expected 14 landmark types")
    town_names = towns.set_index("town_id")["town_name"].to_dict()
    locality_choices = {town_id: list(group[["locality_id", "locality_name"]].itertuples(index=False, name=None))
                        for town_id, group in localities.groupby("town_id")}
    valid_pincodes = set(localities["pincode"].dropna())
    print("Assumption landmark mappings:", ASSUMPTION_TYPE_MAP)

    LABEL_DIR.mkdir(exist_ok=True)
    all_reviews = []
    for split_name, input_file in INPUT_FILES.items():
        table = pd.read_csv(input_file, dtype=str).fillna("")
        labels, reviews = process_table(table, town_names, locality_choices, valid_pincodes)
        labels.to_csv(LABEL_DIR / f"{split_name}_bio.csv", index=False)
        all_reviews.append(reviews)
        in_region = table[table["town_id"] != "OUT"]
        labelled_ids = set(labels.loc[labels["tag"].str.endswith("LOCALITY"), "address_id"])
        share = in_region["address_id"].isin(labelled_ids).mean() if len(in_region) else 0
        print(f"{split_name}: {share:.1%} of in-region rows have a locality label")
        quality_counts = reviews["label_quality"].value_counts().reindex(["sure", "fuzzy", "no_locality", "conflict"], fill_value=0)
        quality_report = pd.DataFrame({"count": quality_counts, "share": (quality_counts / len(reviews)).map("{:.1%}".format)})
        print(quality_report.to_string())

    review_pool = pd.concat(all_reviews, ignore_index=True)
    in_region = review_pool[review_pool["town_id"] != "OUT"]
    sample = in_region.sample(n=min(150, len(in_region)), random_state=42)
    extras = pd.concat([review_pool[review_pool["label_quality"] == quality].head(50) for quality in ("conflict", "fuzzy")])
    review = pd.concat([sample, extras]).drop_duplicates("address_id")
    review.to_csv(LABEL_DIR / "review_sample.csv", index=False)
    print(f"review_sample.csv: {len(review):,} addresses")
    print("Test data was not read or modified.")


if __name__ == "__main__":
    main()
