"""Run MuRIL inference and coordinate fallback on validation addresses only."""

from pathlib import Path
import importlib.util
import re

import pandas as pd
import torch
from rapidfuzz import fuzz, process
from transformers import AutoModelForTokenClassification, AutoTokenizer


PROJECT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT / "models" / "ner_v1"
OUTPUT = PROJECT / "outputs" / "val_predictions.csv"
LABELS = ["O", "B-TOWN", "I-TOWN", "B-LANDMARK", "I-LANDMARK", "B-LOCALITY", "I-LOCALITY", "B-RELATION", "I-RELATION", "B-PINCODE"]
TYPE_MAP = {"mosque": "masjid", "school": "govt_school", "dairy": "milk_dairy", "petrol pump": "petrol_bunk", "wedding hall": "community_hall"}


def load_transliterator():
    spec = importlib.util.spec_from_file_location("prepare", PROJECT / "code_files" / "01_prepare_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.transliterate_indic


def spans(words, tags, kind):
    found, current = [], []
    for word, tag in zip(words, tags):
        if tag == f"B-{kind}":
            if current: found.append(" ".join(current))
            current = [word]
        elif tag == f"I-{kind}" and current:
            current.append(word)
        else:
            if current: found.append(" ".join(current)); current = []
    if current: found.append(" ".join(current))
    return found


def nearest_type(text, types):
    generic = text.replace("_", " ")
    match = process.extractOne(generic, types, scorer=fuzz.token_set_ratio)
    return match[0] if match and match[1] >= 75 else None


def main():
    validation = pd.read_csv(PROJECT / "data" / "labels" / "validation_bio.csv", dtype=str).fillna("")
    addresses = pd.read_csv(PROJECT / "data" / "addresses.csv", dtype=str).fillna("")
    towns = pd.read_csv(PROJECT / "data" / "towns.csv", dtype=str).fillna("")
    localities = pd.read_csv(PROJECT / "data" / "localities.csv", dtype=str).fillna("")
    pois = pd.read_csv(PROJECT / "data" / "landmarks_poi.csv", dtype=str).fillna("")
    for name, table in {"validation_bio": validation, "addresses": addresses, "towns": towns, "localities": localities, "landmarks_poi": pois}.items(): print(name, list(table.columns))
    print("14 landmark types:", sorted(pois.landmark_type.unique()))
    print("ASSUMPTION type mapping:", TYPE_MAP)
    ids = validation.address_id.drop_duplicates()
    raw = addresses[addresses.address_id.isin(ids)][["address_id", "address_text"]]
    if len(raw) != len(ids): raise ValueError("Validation address text is missing")
    transliterate = load_transliterator()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR).eval()
    town_xy = localities.assign(centroid_x=lambda d: d.centroid_x.astype(float), centroid_y=lambda d: d.centroid_y.astype(float)).groupby("town_id")[["centroid_x", "centroid_y"]].mean()
    rows = []
    for record in raw.itertuples(index=False):
        text = transliterate(record.address_text.strip().lower()); words = re.findall(r"\d+(?:st|nd|rd|th)\b|\d+|[a-z]+", text)
        enc = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad(): pred = model(**enc).logits.argmax(-1)[0].tolist()
        tags, previous = [], None
        for word_id, label_id in zip(enc.word_ids(), pred):
            if word_id is not None and word_id != previous: tags.append(LABELS[label_id])
            previous = word_id
        town_span = (spans(words, tags, "TOWN") or [""])[0]
        town_match = process.extractOne(town_span, towns.town_name.tolist(), scorer=fuzz.token_set_ratio) if town_span else None
        town_id = towns.iloc[town_match[2]].town_id if town_match and town_match[1] >= 80 and "village" not in text else "OUT"
        locality_span = (spans(words, tags, "LOCALITY") or [""])[0]
        candidates = localities[localities.town_id.eq(town_id)] if town_id != "OUT" else localities.iloc[0:0]
        locality_match = process.extractOne(locality_span, candidates.locality_name.tolist(), scorer=fuzz.token_set_ratio) if locality_span else None
        locality = candidates.iloc[locality_match[2]] if locality_match and locality_match[1] >= 80 else None
        landmark_span = (spans(words, tags, "LANDMARK") or [""])[0]
        landmark_type = nearest_type(landmark_span, pois.landmark_type.tolist()) if landmark_span else None
        poi = None
        if locality is not None and landmark_type:
            options = pois[(pois.town_id == town_id) & (pois.landmark_type == landmark_type)].copy()
            if len(options):
                options["d"] = (options.x.astype(float)-float(locality.centroid_x))**2 + (options.y.astype(float)-float(locality.centroid_y))**2
                poi = options.loc[options.d.idxmin()]
        if poi is not None: x, y, level = poi.x, poi.y, "landmark"
        elif locality is not None: x, y, level = locality.centroid_x, locality.centroid_y, "locality"
        elif town_id != "OUT": x, y, level = town_xy.loc[town_id, "centroid_x"], town_xy.loc[town_id, "centroid_y"], "town"
        else: x, y, level = "", "", "out"
        rows.append({"address_id": record.address_id, "town_id": town_id, "locality_id": locality.locality_id if locality is not None else "", "landmark_type": landmark_type or "", "x": x, "y": y, "level_used": level})
    out = pd.DataFrame(rows); OUTPUT.parent.mkdir(exist_ok=True); out.to_csv(OUTPUT, index=False)
    print(out.level_used.value_counts(normalize=True).to_string())
    print(f"wrote: {OUTPUT}")


if __name__ == "__main__": main()
