"""Run the fixed final pipeline on the 90 no-OUT holdout addresses only."""
from pathlib import Path
import importlib.util
import re

import pandas as pd
import torch
from rapidfuzz import fuzz, process, utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForTokenClassification, AutoTokenizer


P = Path(__file__).resolve().parents[1]
TOK = re.compile(r"\d+(?:st|nd|rd|th)\b|\d+|[a-z]+")
LABELS = ["O", "B-TOWN", "I-TOWN", "B-LANDMARK", "I-LANDMARK", "B-LOCALITY", "I-LOCALITY", "B-RELATION", "I-RELATION", "B-PINCODE"]


def spans(words, tags, kind):
    output, current = [], []
    for word, tag in zip(words, tags):
        if tag == f"B-{kind}":
            if current: output.append(" ".join(current))
            current = [word]
        elif tag == f"I-{kind}" and current:
            current.append(word)
        else:
            if current: output.append(" ".join(current)); current = []
    if current: output.append(" ".join(current))
    return output


def without_pincode(text):
    return " ".join(word for word in text.split() if not re.fullmatch(r"\d{5,6}", word))


def main():
    holdout = pd.read_csv(P / "temp_code" / "holdout_ids_noout.csv", dtype=str)
    addresses = pd.read_csv(P / "data" / "addresses.csv", dtype=str).fillna("")
    train = pd.read_csv(P / "data" / "labels" / "train_bio_merged_v2_noout.csv", dtype=str).fillna("")
    towns = pd.read_csv(P / "data" / "towns.csv", dtype=str)
    localities = pd.read_csv(P / "data" / "localities.csv", dtype=str)
    pois = pd.read_csv(P / "data" / "landmarks_poi.csv", dtype=str)
    for name, table in [("holdout", holdout), ("addresses", addresses), ("train", train), ("towns", towns), ("localities", localities), ("pois", pois)]:
        print(name, list(table.columns))
    raw = addresses[addresses.address_id.isin(set(holdout.address_id))][["address_id", "address_text"]]
    if len(raw) != len(holdout): raise ValueError("Holdout address is missing")

    assignment = {}
    for poi in pois.itertuples(index=False):
        candidates = localities[localities.town_id.eq(poi.town_id)].copy()
        candidates["distance"] = (candidates.centroid_x.astype(float) - float(poi.x)) ** 2 + (candidates.centroid_y.astype(float) - float(poi.y)) ** 2
        assignment[poi.poi_id] = candidates.loc[candidates.distance.idxmin(), "locality_id"]

    examples = []
    for _, group in train.groupby("address_id", sort=False):
        group = group.sort_values("word_index", key=lambda values: values.astype(int))
        landmark = spans(group.word.tolist(), group.tag.tolist(), "LANDMARK")
        types = [value for value in group.landmark_type if value and value != "temple_unknown"]
        if landmark and types: examples.append((landmark[0], pd.Series(types).mode().iloc[0]))
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=1)
    classifier = LogisticRegression(max_iter=1000, random_state=42).fit(vectorizer.fit_transform([x[0] for x in examples]), [x[1] for x in examples])

    spec = importlib.util.spec_from_file_location("prepare", P / "code_files" / "01_prepare_data.py")
    prep = importlib.util.module_from_spec(spec); spec.loader.exec_module(prep)
    tokenizer = AutoTokenizer.from_pretrained(P / "models" / "ner_v2_best")
    model = AutoModelForTokenClassification.from_pretrained(P / "models" / "ner_v2_best").eval()
    town_centroids = localities.astype({"centroid_x": float, "centroid_y": float}).groupby("town_id")[["centroid_x", "centroid_y"]].mean()
    rows = []
    for row in raw.itertuples(index=False):
        words = TOK.findall(prep.transliterate_indic(row.address_text.lower()))
        encoded = tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad(): predictions = model(**encoded).logits.argmax(-1)[0].tolist()
        tags, previous = [], None
        for word_id, prediction in zip(encoded.word_ids(), predictions):
            if word_id is not None and word_id != previous: tags.append(LABELS[prediction])
            previous = word_id
        town_span = without_pincode((spans(words, tags, "TOWN") or [""])[0])
        locality_span = without_pincode((spans(words, tags, "LOCALITY") or [""])[0])
        landmark_span = without_pincode((spans(words, tags, "LANDMARK") or [""])[0])
        pincodes = [word for word in words if re.fullmatch(r"\d{5,6}", word)]
        match = process.extractOne(town_span, towns.town_name.tolist(), scorer=fuzz.WRatio, processor=utils.default_process) if town_span else None
        town_id = towns.iloc[match[2]].town_id if match and match[1] >= 85 else ""
        if not town_id and pincodes:
            found = localities[localities.pincode.isin(pincodes)].town_id.unique()
            town_id = found[0] if len(found) == 1 else "UNKNOWN"
        if not town_id: town_id = "UNKNOWN"
        candidates = localities[localities.town_id.eq(town_id)]
        if pincodes: candidates = candidates[candidates.pincode.isin(pincodes)]
        local_match = process.extractOne(locality_span, candidates.locality_name.tolist(), scorer=fuzz.WRatio, processor=utils.default_process) if locality_span and len(candidates) else None
        locality = candidates.iloc[local_match[2]] if local_match and local_match[1] >= 85 else None
        if locality is None and len(candidates) == 1: locality = candidates.iloc[0]
        landmark_type = poi_name = ""
        if landmark_span and locality is not None:
            probabilities = classifier.predict_proba(vectorizer.transform([landmark_span]))[0]
            options = pois[(pois.town_id.eq(town_id)) & (pois.poi_id.map(assignment).eq(locality.locality_id))]
            for index in probabilities.argsort()[::-1]:
                if classifier.classes_[index] in set(options.landmark_type) and probabilities[index] >= .5:
                    landmark_type = classifier.classes_[index]
                    choices = options[options.landmark_type.eq(landmark_type)].copy()
                    choices["distance"] = (choices.x.astype(float) - float(locality.centroid_x)) ** 2 + (choices.y.astype(float) - float(locality.centroid_y)) ** 2
                    selected = choices.loc[choices.distance.idxmin()]
                    poi_name, x, y, level = selected['name'], selected.x, selected.y, "landmark"
                    break
        if locality is not None and not landmark_type:
            x, y, level = locality.centroid_x, locality.centroid_y, "locality"
        elif locality is None and town_id != "UNKNOWN":
            x, y, level = town_centroids.loc[town_id, "centroid_x"], town_centroids.loc[town_id, "centroid_y"], "town"
        elif locality is None:
            x, y, level = "", "", "town"
        rows.append({"address_id": row.address_id, "raw_text": row.address_text, "landmark_span": landmark_span, "predicted_locality_name": locality.locality_name if locality is not None else "", "predicted_landmark_type": landmark_type, "poi_name": poi_name, "level_used": level, "x": x, "y": y})
    result = pd.DataFrame(rows)
    result.to_csv(P / "outputs" / "holdout_predictions.csv", index=False)
    print(result.level_used.value_counts().to_string())
    print(f"wrote {len(result)} holdout predictions")


if __name__ == "__main__": main()
