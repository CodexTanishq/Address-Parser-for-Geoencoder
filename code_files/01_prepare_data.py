"""Prepare train/validation addresses without changing the held-out test split."""

from collections import Counter
from pathlib import Path
import random
import re
import sys
import unicodedata

import pandas as pd
from indic_transliteration import sanscript
from rapidfuzz.distance import Levenshtein


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = DATA_DIR / "cleaned"
ADDRESS_FILE = DATA_DIR / "addresses.csv"
SPLIT_FILE = DATA_DIR / "splits.csv" if (DATA_DIR / "splits.csv").exists() else DATA_DIR / "split.csv"
ACTIVE_SPLITS = ("train", "validation")  # Test remains untouched until approval.
ALL_SPLITS = ("train", "validation", "test")

# Supplied by the user or English names of the requested groups.
GENERAL_SEEDS = {
    "relation:near": ["near", "nr", "hattira", "samipa", "bali", "nazdik", "paas"],
    "relation:opposite": ["opposite", "opp"], "relation:behind": ["behind"],
    "relation:beside": ["beside"], "relation:next_to": ["next", "nextto"],
    "landmark:temple": ["temple", "mandir"], "landmark:mosque": ["mosque", "masjid"],
    "landmark:church": ["church"], "landmark:school": ["school", "shaale"],
    "landmark:bus_stop": ["bus", "stop", "busstop"], "landmark:park": ["park"],
    "landmark:post_office": ["post", "office", "postoffice"],
    "landmark:medical_store": ["medical", "medicine", "pharmacy"],
    "landmark:ration_shop": ["ration"], "landmark:dairy": ["dairy", "milk"],
    "landmark:water_tank": ["water", "tank", "watertank"],
    "landmark:petrol_pump": ["petrol", "pump", "petrolpump", "fuel", "bunk"],
    "landmark:wedding_hall": ["mantapa"],
}

# Approved dataset words. They enter the seed list only when seen in TRAIN.
APPROVED_TRAIN_WORD_GROUPS = {
    "charcha": "landmark:church", "tamki": "landmark:water_tank", "dakaghara": "landmark:post_office",
    "deyari": "landmark:dairy", "dudha": "landmark:dairy", "parka": "landmark:park",
    "pasa": "relation:near", "piche": "relation:behind", "petrola": "landmark:petrol_pump",
    "bagala": "relation:beside", "basa": "landmark:bus_stop", "mamdira": "landmark:temple",
    "masjida": "landmark:mosque", "medikala": "landmark:medical_store", "rashana": "landmark:ration_shop",
    "samane": "relation:opposite", "skula": "landmark:school", "stopa": "landmark:bus_stop",
    "amche": "landmark:post_office", "udyanavana": "landmark:park", "eduru": "relation:opposite",
    "charch": "landmark:church", "tyamk": "landmark:water_tank", "dairi": "landmark:dairy",
    "devasthana": "landmark:temple", "nildana": "landmark:bus_stop", "nyayabele": "landmark:ration_shop",
    "pakka": "relation:beside", "petrol": "landmark:petrol_pump", "bamk": "landmark:petrol_pump",
    "bas": "landmark:bus_stop", "mamtapa": "landmark:wedding_hall", "masidi": "landmark:mosque",
    "medikal": "landmark:medical_store", "shale": "landmark:school", "hattira": "relation:near",
    "halina": "landmark:dairy", "himde": "relation:behind",
}

FILLER_WORDS = {"ke", "ki", "ka", "mem", "mein", "me", "se", "ko", "no", "ku", "ge", "alli", "inda", "kevala", "the", "of", "and", "to"}
USER_SPELLINGS = {"nr", "opp", "shaale", "mandir", "masjid", "hattira", "samipa", "bali", "nazdik", "paas"}
ASSUMPTION_WORDS = sorted(({word for words in GENERAL_SEEDS.values() for word in words} - USER_SPELLINGS) | (FILLER_WORDS - {"ke", "ki", "ka", "mem", "mein", "me", "se", "ko"}))

DEVANAGARI = re.compile(r"[\u0900-\u097F]+")
KANNADA = re.compile(r"[\u0C80-\u0CFF]+")
INDIC = re.compile(r"[\u0900-\u097F\u0C80-\u0CFF]")
TOKEN = re.compile(r"\d+(?:st|nd|rd|th)\b|[a-z]+")
DOUBLE_LETTER = re.compile(r"(.)\1+")
VOWELS = set("aeiou")


def remove_accents(text: str) -> str:
    return "".join(char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn")


def transliterate_indic(text: str) -> str:
    """Apply Hindi fixes before transliterating Devanagari and Kannada only."""
    text = text.replace("नं.", "no").replace("नं", "no").replace("ॉ", "ो").replace("ॅ", "े")
    text = DEVANAGARI.sub(lambda m: sanscript.transliterate(m.group(), sanscript.DEVANAGARI, sanscript.ITRANS), text)
    text = KANNADA.sub(lambda m: sanscript.transliterate(m.group(), sanscript.KANNADA, sanscript.ITRANS), text)
    return remove_accents(text).lower()


def loose_key(word: str) -> str:
    word = DOUBLE_LETTER.sub(r"\1", word.lower())
    return word if len(word) < 4 else word[0] + "".join(letter for letter in word[1:] if letter not in VOWELS)


def build_seed_groups(train_texts: pd.Series) -> dict[str, list[str]]:
    train_words = {word for text in train_texts for word in TOKEN.findall(text)}
    groups = {group: list(words) for group, words in GENERAL_SEEDS.items()}
    for word, group in APPROVED_TRAIN_WORD_GROUPS.items():
        if word in train_words:
            groups[group].append(word)
    return {group: sorted(set(words)) for group, words in groups.items()}


def matched_group(word: str, seed_groups: dict[str, list[str]]) -> str | None:
    if word in FILLER_WORDS or any(char.isdigit() for char in word):
        return None
    maximum_distance = 0 if len(word) <= 3 else 1 if len(word) <= 6 else 2
    key = loose_key(word)
    candidates = []
    for group, spellings in seed_groups.items():
        distance = min(Levenshtein.distance(word, spelling) for spelling in spellings)
        key_match = len(word) >= 4 and any(key == loose_key(spelling) for spelling in spellings)
        if distance <= maximum_distance or key_match:
            candidates.append((distance, group))
    return min(candidates)[1] if candidates else None


def phrase_matches(words: list[str]) -> list[tuple[int, str]]:
    """Kalyana alone is never a landmark; only approved wedding-hall phrases are."""
    return [(index, "landmark:wedding_hall") for index in range(len(words) - 1)
            if f"{words[index]} {words[index + 1]}" in {"barata ghara", "kalyana mamtapa", "kalyana mantapa"}]


def analyse_text(text: str, seed_groups: dict[str, list[str]], unknown_words: Counter | None = None) -> tuple[str, bool, Counter]:
    words = TOKEN.findall(text)
    matches = [(index, matched_group(word, seed_groups)) for index, word in enumerate(words)]
    matches = [(index, group) for index, group in matches if group]
    matches.extend(phrase_matches(words))
    matches.sort()
    if unknown_words is not None:
        for word in words:
            if word not in FILLER_WORDS and not any(char.isdigit() for char in word) and not matched_group(word, seed_groups):
                unknown_words[word] += 1
    relations = [index for index, group in matches if group.startswith("relation:")]
    landmarks = [(index, group) for index, group in matches if group.startswith("landmark:")]
    has_relation = any(abs(landmark - relation) <= 3 for landmark, _ in landmarks for relation in relations)
    targets = {"landmark:school", "landmark:mosque", "landmark:post_office", "landmark:park"}
    misses = Counter(group for landmark, group in landmarks if group in targets and not any(abs(landmark - relation) <= 3 for relation in relations))
    return ";".join(dict.fromkeys(group for _, group in matches)), has_relation, misses


def enrich_table(table: pd.DataFrame, seed_groups: dict[str, list[str]], unknown_words: Counter) -> tuple[pd.DataFrame, Counter]:
    cleaned = table.copy()
    cleaned["address_text"] = cleaned["address_text"].fillna("").astype(str).str.strip().str.lower().map(transliterate_indic)
    remaining = cleaned["address_text"].map(lambda text: bool(INDIC.search(text)))
    print(f"{len(cleaned):,} rows processed; {int(remaining.sum()):,} rows still contain Hindi/Kannada letters")
    if remaining.any():
        raise ValueError("Hindi or Kannada letters remain after transliteration")
    results = [analyse_text(text, seed_groups, unknown_words) for text in cleaned["address_text"]]
    cleaned["matched_groups"] = [groups for groups, _, _ in results]
    cleaned["landmark_has_relation"] = [flag for _, flag, _ in results]
    misses = Counter()
    for _, _, item in results:
        misses.update(item)
    return cleaned, misses


def build_stress_set(train: pd.DataFrame, seed_groups: dict[str, list[str]]) -> float:
    """Perturb train only; the generated examples never update seed words."""
    replacements = {"near": "nr", "hattira": "bali", "samane": "opp", "opposite": "opp", "school": "shaale", "temple": "mandir", "mosque": "masjid", "pasa": "paas", "petrola": "petrol", "mamdira": "mandir"}
    rng = random.Random(42)
    sample = train.sample(n=min(100, len(train)), random_state=42).copy()
    stressed_texts, found, expected = [], 0, 0
    for text in sample["address_text"]:
        words, stressed, changed = TOKEN.findall(text), text, False
        for word in words:
            if word in replacements:
                stressed = re.sub(rf"\b{re.escape(word)}\b", replacements[word], stressed, count=1)
                changed = True
                break
        if not changed:
            candidates = [word for word in words if len(word) > 4 and word not in FILLER_WORDS]
            if candidates:
                word = rng.choice(candidates)
                typo = word[:-1] if len(word) > 5 else word[0] + word[2:]
                stressed = re.sub(rf"\b{re.escape(word)}\b", typo, stressed, count=1)
        base, _, _ = analyse_text(text, seed_groups)
        stressed_groups, _, _ = analyse_text(stressed, seed_groups)
        base_set, stress_set = set(filter(None, base.split(";"))), set(filter(None, stressed_groups.split(";")))
        expected += len(base_set)
        found += len(base_set & stress_set)
        stressed_texts.append(stressed)
    sample["stressed_address_text"] = stressed_texts
    sample.to_csv(DATA_DIR / "stress_set.csv", index=False)
    return found / expected if expected else 1.0


def report_unseen_indic_words(joined: pd.DataFrame) -> None:
    def script_words(texts: pd.Series) -> set[str]:
        return {word for text in texts.fillna("") for word in DEVANAGARI.findall(text) + KANNADA.findall(text)}
    train_words = script_words(joined.loc[joined["split"] == "train", "address_text"])
    for split_name in ("validation", "test"):
        unseen = sorted(script_words(joined.loc[joined["split"] == split_name, "address_text"]) - train_words)
        print(f"Hindi/Kannada words in {split_name} not in train ({len(unseen)}): {unseen}")


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    splits = pd.read_csv(SPLIT_FILE, dtype=str)
    print("split columns:", list(splits.columns))
    print(splits.head().to_string(index=False))
    if not {"account_id", "split"}.issubset(splits.columns) or set(splits["split"].dropna()) - set(ALL_SPLITS):
        raise ValueError("invalid split file")
    if splits.groupby("account_id")["split"].nunique().gt(1).any():
        raise ValueError("an account_id is assigned to more than one split")
    splits = splits.drop_duplicates("account_id")
    addresses = pd.read_csv(ADDRESS_FILE, dtype={"address_id": str, "town_id": str, "account_id": str})
    required = {"address_id", "town_id", "address_text", "account_id"}
    if not required.issubset(addresses.columns):
        raise ValueError("addresses.csv is missing a required column")
    joined = addresses[["address_id", "town_id", "address_text", "account_id"]].merge(splits, on="account_id", how="left", validate="many_to_one")
    if joined["split"].isna().any() or joined["address_id"].duplicated().any():
        raise ValueError("addresses must have one split assignment each")
    for field in ("account_id", "address_id"):
        sets = {name: set(joined.loc[joined["split"] == name, field]) for name in ALL_SPLITS}
        overlap = sum(len(sets[left] & sets[right]) for index, left in enumerate(ALL_SPLITS) for right in ALL_SPLITS[index + 1:])
        print(f"shared {field} between train/validation/test: {overlap}")
        if overlap:
            raise ValueError(f"{field} appears in more than one split")
    report_unseen_indic_words(joined)
    train_raw = joined.loc[joined["split"] == "train", ["address_id", "town_id", "address_text"]]
    train_for_seeds = train_raw["address_text"].fillna("").astype(str).str.strip().str.lower().map(transliterate_indic)
    seed_groups = build_seed_groups(train_for_seeds)
    print("General-knowledge assumption words:", ASSUMPTION_WORDS)
    OUTPUT_DIR.mkdir(exist_ok=True)
    unknown_words, landmark_misses, examples = Counter(), Counter(), []
    for split_name in ACTIVE_SPLITS:
        table = joined.loc[joined["split"] == split_name, ["address_id", "town_id", "address_text"]]
        before = table["address_text"].copy()
        cleaned, misses = enrich_table(table, seed_groups, unknown_words)
        cleaned.to_csv(OUTPUT_DIR / f"{split_name}_cleaned.csv", index=False)
        landmark_misses.update(misses)
        examples.extend((original, translated) for original, translated in zip(before, cleaned["address_text"]) if DEVANAGARI.search(original) or KANNADA.search(original))
    pd.DataFrame(unknown_words.most_common(), columns=["word", "count"]).to_csv(OUTPUT_DIR / "unknown_words.csv", index=False)
    coverage = build_stress_set(pd.read_csv(OUTPUT_DIR / "train_cleaned.csv", dtype=str), seed_groups)
    print("Landmark misses without a relation within 3 words:")
    for group in ("landmark:school", "landmark:mosque", "landmark:post_office", "landmark:park"):
        print(f"  {group}: {landmark_misses[group]}")
    print(f"Stress-set group retention: {coverage:.1%}")
    random.Random(42).shuffle(examples)
    print("\n20 random before/after Hindi/Kannada examples:")
    for before, after in examples[:20]:
        print(f"BEFORE: {before}\nAFTER:  {after}\n")
    print("Test split was read only for leakage/unseen-word reporting; no test output was changed.")


if __name__ == "__main__":
    main()
