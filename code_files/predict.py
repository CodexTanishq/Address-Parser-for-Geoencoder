"""Final address inference: NER, landmark type, POI, and area fallback."""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
from pathlib import Path

import pandas as pd
import torch
from rapidfuzz import fuzz, process, utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoModelForTokenClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
LABELS = ["O", "B-TOWN", "I-TOWN", "B-LANDMARK", "I-LANDMARK", "B-LOCALITY", "I-LOCALITY", "B-RELATION", "I-RELATION", "B-PINCODE"]
TOKEN_RE = re.compile(r"\d+(?:st|nd|rd|th)\b|\d+|[a-z]+")
PIN_RE = re.compile(r"\b\d{5,6}\b")
CHOICE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


def require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def entity_spans(words: list[str], tags: list[str], entity: str) -> list[list[tuple[int, str]]]:
    found, current = [], []
    for index, (word, tag) in enumerate(zip(words, tags)):
        if tag == f"B-{entity}":
            if current:
                found.append(current)
            current = [(index, word)]
        elif tag == f"I-{entity}" and current:
            current.append((index, word))
        else:
            if current:
                found.append(current)
                current = []
    if current:
        found.append(current)
    return found


def train_classifier(train: pd.DataFrame) -> tuple[TfidfVectorizer, LogisticRegression]:
    examples: list[tuple[str, str]] = []
    for _, group in train.groupby("address_id", sort=False):
        group = group.sort_values("word_index", key=lambda value: value.astype(int))
        spans = entity_spans(group.word.tolist(), group.tag.tolist(), "LANDMARK")
        types = [value for value in group.landmark_type.tolist() if value and value != "temple_unknown"]
        if spans and types:
            examples.append((" ".join(word for _, word in spans[0]), pd.Series(types).mode().iloc[0]))
    if not examples:
        raise ValueError("No train landmark spans are available for the classifier")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(vectorizer.fit_transform([span for span, _ in examples]), [kind for _, kind in examples])
    return vectorizer, classifier


class QwenChoice:
    """Constrained likelihood scorer used by the final 11_coordinates flow."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(CHOICE_MODEL, local_files_only=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            CHOICE_MODEL, local_files_only=True, torch_dtype=dtype
        ).to(self.device).eval()

    def scores(self, span: str, type_names: list[str]) -> torch.Tensor:
        text = (
            "This Indian address landmark may be written in Hindi or Kannada in English letters. "
            f"Which type is it? Landmark: {span}. Type:\n"
            f"Allowed types: {', '.join(type_names)}\nType:"
        )
        prompt = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True
        )
        prefix = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        values = []
        for type_name in type_names:
            choice = self.tokenizer(" " + type_name, add_special_tokens=False, return_tensors="pt").input_ids[0]
            input_ids = torch.cat([prefix, choice]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=input_ids).logits[0]
            log_prob = torch.log_softmax(logits[len(prefix) - 1:-1], dim=-1)
            values.append(log_prob.gather(1, choice.to(self.device).unsqueeze(1)).sum())
        return torch.stack(values).detach().cpu()


class Predictor:
    def __init__(self) -> None:
        self.addresses = pd.read_csv(DATA / "addresses.csv", dtype=str).fillna("")
        self.train = pd.read_csv(DATA / "labels" / "train_bio_merged_v2_noout.csv", dtype=str).fillna("")
        self.towns = pd.read_csv(DATA / "towns.csv", dtype=str).fillna("")
        self.localities = pd.read_csv(DATA / "localities.csv", dtype=str).fillna("")
        self.pois = pd.read_csv(DATA / "landmarks_poi.csv", dtype=str).fillna("")
        self.baseline = pd.read_csv(DATA / "baseline_geocodes.csv", dtype=str).fillna("")
        for name, frame, columns in [
            ("addresses", self.addresses, ["address_id", "address_text"]),
            ("train labels", self.train, ["address_id", "word_index", "word", "tag", "landmark_type"]),
            ("towns", self.towns, ["town_id", "town_name"]),
            ("localities", self.localities, ["locality_id", "locality_name", "town_id", "pincode", "centroid_x", "centroid_y"]),
            ("landmarks", self.pois, ["poi_id", "town_id", "landmark_type", "name", "x", "y"]),
            ("baseline geocodes", self.baseline, ["address_id", "geocoder_x", "geocoder_y", "precision"]),
        ]:
            require(frame, columns, name)
        for column in ["centroid_x", "centroid_y"]:
            self.localities[column] = pd.to_numeric(self.localities[column], errors="raise")
        for column in ["x", "y"]:
            self.pois[column] = pd.to_numeric(self.pois[column], errors="raise")
        for column in ["geocoder_x", "geocoder_y"]:
            self.baseline[column] = pd.to_numeric(self.baseline[column], errors="raise")

        spec = importlib.util.spec_from_file_location("prepare", ROOT / "code_files" / "01_prepare_data.py")
        self.prepare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.prepare)
        self.ner_tokenizer = AutoTokenizer.from_pretrained(ROOT / "models" / "ner_v2_best")
        self.ner_model = AutoModelForTokenClassification.from_pretrained(ROOT / "models" / "ner_v2_best").eval()
        self.vectorizer, self.classifier = train_classifier(self.train)
        self.types = sorted(self.pois.landmark_type.unique())
        if set(self.classifier.classes_) - set(self.types):
            raise ValueError("Classifier type is absent from landmarks_poi.csv")
        self.pois = self._assign_catchments(self.pois)
        self.town_centres = self.localities.groupby("town_id", as_index=False)[["centroid_x", "centroid_y"]].mean()
        self.choice_model: QwenChoice | None = None
        self.content_free: torch.Tensor | None = None
        self.choice_cache: dict[str, tuple[str, str]] = {}

    def _assign_catchments(self, poi: pd.DataFrame) -> pd.DataFrame:
        assigned = []
        for item in poi.itertuples(index=False):
            choices = self.localities[self.localities.town_id.eq(item.town_id)]
            if choices.empty:
                raise ValueError(f"No locality exists in POI town {item.town_id}")
            distance2 = (choices.centroid_x - float(item.x)) ** 2 + (choices.centroid_y - float(item.y)) ** 2
            assigned.append(choices.loc[distance2.idxmin(), "locality_id"])
        result = poi.copy()
        result["catchment_locality_id"] = assigned
        return result

    def _ner(self, raw: str) -> tuple[list[str], list[str], list[float]]:
        words = TOKEN_RE.findall(self.prepare.transliterate_indic(raw.lower()))
        if not words:
            return [], [], []
        encoded = self.ner_tokenizer(words, is_split_into_words=True, return_tensors="pt", truncation=True, max_length=128)
        with torch.no_grad():
            logits = self.ner_model(**encoded).logits[0]
            ids = logits.argmax(-1).tolist()
            confidence = logits.softmax(-1).max(-1).values.tolist()
        tags, values, previous = [], [], None
        for word_id, label, value in zip(encoded.word_ids(), ids, confidence):
            if word_id is not None and word_id != previous:
                tags.append(LABELS[label])
                values.append(float(value))
            previous = word_id
        return words, tags, values

    def _area(self, town_id: str, locality_id: str, pin: str) -> tuple[str, object, object]:
        locality = self.localities[self.localities.locality_id.eq(locality_id)] if locality_id else self.localities.iloc[0:0]
        if not town_id:
            return "unknown_town", "", ""
        if not locality.empty:
            return "locality", locality.iloc[0].centroid_x, locality.iloc[0].centroid_y
        pin_locs = self.localities[(self.localities.town_id.eq(town_id)) & (self.localities.pincode.eq(pin))]
        if not pin_locs.empty:
            return "pincode_area", pin_locs.centroid_x.mean(), pin_locs.centroid_y.mean()
        centre = self.town_centres[self.town_centres.town_id.eq(town_id)]
        if centre.empty:
            return "unknown_town", "", ""
        return "town", centre.iloc[0].centroid_x, centre.iloc[0].centroid_y

    def _final_type(self, span: str) -> tuple[str, float, str]:
        probabilities = self.classifier.predict_proba(self.vectorizer.transform([span]))[0]
        top = int(probabilities.argmax())
        classifier_type = str(self.classifier.classes_[top])
        classifier_probability = float(probabilities[top])
        if classifier_probability >= 0.5:
            return classifier_type, classifier_probability, "classifier"
        if span in self.choice_cache:
            final_type, source = self.choice_cache[span]
            return final_type, classifier_probability, source
        distribution = torch.zeros(len(self.types), dtype=torch.float32)
        for label, value in zip(self.classifier.classes_, probabilities):
            distribution[self.types.index(label)] = float(value)
        if self.choice_model is None:
            self.choice_model = QwenChoice()
        if self.content_free is None:
            self.content_free = self.choice_model.scores("N/A", self.types)
        qwen = torch.softmax(self.choice_model.scores(span, self.types) - self.content_free, dim=0)
        final_type = self.types[int(torch.argmax((distribution + qwen) / 2))]
        self.choice_cache[span] = (final_type, "avg_llm")
        return final_type, classifier_probability, "avg_llm"

    def _candidates(self, town_id: str, locality_id: str, pin: str, landmark_type: str) -> tuple[pd.DataFrame, int]:
        base = self.pois[(self.pois.town_id.eq(town_id)) & (self.pois.landmark_type.eq(landmark_type))]
        if base.empty:
            return base, 4
        if locality_id:
            one = base[base.catchment_locality_id.eq(locality_id)]
            if not one.empty:
                return one, 1
            loc_row = self.localities[self.localities.locality_id.eq(locality_id)]
            if not loc_row.empty:
                same_pin = self.localities[(self.localities.town_id.eq(town_id)) & (self.localities.pincode.eq(loc_row.iloc[0].pincode))]
                two = base[base.catchment_locality_id.isin(same_pin.locality_id)]
                if not two.empty:
                    return two, 2
        if not locality_id and pin:
            pin_locs = self.localities[(self.localities.town_id.eq(town_id)) & (self.localities.pincode.eq(pin))]
            three = base[base.catchment_locality_id.isin(pin_locs.locality_id)]
            if not three.empty:
                return three, 3
        return base, 4

    def predict(self, raw: str, address_id: str = "") -> dict:
        empty = {"address_id": address_id, "address": raw, "town_id": "", "town_name": "", "locality_id": "", "locality_name": "", "locality_x": "", "locality_y": "", "area_level": "", "area_x": "", "area_y": "", "ref_point_source": "", "landmarks": "NO LANDMARK IN ADDRESS"}
        if not raw or not raw.strip():
            return empty
        words, tags, confidence = self._ner(raw)
        town_spans = entity_spans(words, tags, "TOWN")
        town_text = " ".join(word for _, word in town_spans[0]) if town_spans else ""
        match = process.extractOne(town_text, self.towns.town_name.tolist(), scorer=fuzz.WRatio, processor=utils.default_process) if town_text else None
        town_id = self.towns.iloc[match[2]].town_id if match and match[1] >= 85 else ""
        pin = next(iter(PIN_RE.findall(raw)), "")
        if not town_id and pin:
            pin_towns = self.localities[self.localities.pincode.eq(pin)].town_id.unique()
            town_id = pin_towns[0] if len(pin_towns) == 1 else ""
        town_name = self.towns[self.towns.town_id.eq(town_id)].town_name.iloc[0] if town_id else ""
        locality_id = locality_name = ""
        locality_spans = entity_spans(words, tags, "LOCALITY")
        if town_id and locality_spans:
            locality_text = " ".join(word for _, word in locality_spans[0])
            choices = self.localities[self.localities.town_id.eq(town_id)]
            matched = process.extractOne(locality_text, choices.locality_name.tolist(), scorer=fuzz.WRatio, processor=utils.default_process)
            if matched and matched[1] >= 85:
                chosen = choices.iloc[matched[2]]
                locality_id, locality_name = chosen.locality_id, chosen.locality_name
        area_level, area_x, area_y = self._area(town_id, locality_id, pin)
        locality_row = self.localities[self.localities.locality_id.eq(locality_id)] if locality_id else self.localities.iloc[0:0]
        if not locality_row.empty:
            locality_x, locality_y = locality_row.iloc[0].centroid_x, locality_row.iloc[0].centroid_y
        else:
            locality_x = locality_y = ""
        baseline = self.baseline[self.baseline.address_id.eq(address_id)] if address_id else self.baseline.iloc[0:0]
        if not baseline.empty:
            reference_x, reference_y, reference_source = baseline.iloc[0].geocoder_x, baseline.iloc[0].geocoder_y, "baseline"
        elif not locality_row.empty:
            reference_x, reference_y, reference_source = locality_x, locality_y, "locality_centroid"
        else:
            reference_x, reference_y, reference_source = area_x, area_y, "area"
        town_indexes = {index for span in town_spans for index, _ in span}
        landmarks = []
        for span in entity_spans(words, tags, "LANDMARK"):
            clean = [word for index, word in span if index not in town_indexes and not re.search(r"\d", word) and process.extractOne(word, self.towns.town_name.tolist(), scorer=fuzz.WRatio, processor=utils.default_process)[1] < 85]
            if not clean:
                continue
            text = " ".join(clean)
            final_type, probability, source = self._final_type(text)
            candidates, level = self._candidates(town_id, locality_id, pin, final_type) if town_id else (self.pois.iloc[0:0], "")
            item = {"span": text, "type": final_type, "type_source": source, "prob": probability, "poi_id": "", "poi_x": "", "poi_y": "", "n_candidates": int(len(candidates)), "level": level}
            if not candidates.empty and reference_x != "" and reference_y != "":
                distance2 = (candidates.x - float(reference_x)) ** 2 + (candidates.y - float(reference_y)) ** 2
                chosen = candidates.loc[distance2.idxmin()]
                item.update({"poi_id": chosen.poi_id, "poi_x": chosen.x, "poi_y": chosen.y})
            landmarks.append(item)
        return {"address_id": address_id, "address": raw, "town_id": town_id, "town_name": town_name, "locality_id": locality_id, "locality_name": locality_name, "locality_x": locality_x, "locality_y": locality_y, "area_level": area_level, "area_x": area_x, "area_y": area_y, "ref_point_source": reference_source, "landmarks": (landmarks if landmarks else "NO LANDMARK IN ADDRESS")}


def flatten(result: dict) -> dict:
    row = {key: value for key, value in result.items() if key != "landmarks"}
    row["full_json"] = json.dumps(result, ensure_ascii=False, default=str)
    if not isinstance(result["landmarks"], list):
        row["lm1_span"] = "NO LANDMARK IN ADDRESS"
    row["landmarks_json"] = (json.dumps(result["landmarks"], ensure_ascii=False) if isinstance(result["landmarks"], list) else "NO LANDMARK IN ADDRESS")
    for number, landmark in enumerate((result["landmarks"] if isinstance(result["landmarks"], list) else [])[:2], 1):
        for key, value in landmark.items():
            row[f"lm{number}_{key}"] = value
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--address")
    parser.add_argument("--input_csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if bool(args.address is not None) == bool(args.input_csv):
        parser.error("provide exactly one of --address or --input_csv")
    if args.input_csv and not args.output:
        parser.error("--output is required with --input_csv")
    predictor = Predictor()
    if args.address is not None:
        print(json.dumps(predictor.predict(args.address), ensure_ascii=False, default=str))
        return
    source = pd.read_csv(args.input_csv, dtype=str).fillna("")
    if "address_text" not in source.columns and "raw_text" in source.columns:
        source = source.rename(columns={"raw_text": "address_text"})
    require(source, ["address_text"], "input CSV")
    if "address_id" not in source.columns:
        source["address_id"] = ""
    rows = [flatten(predictor.predict(row.address_text, row.address_id)) for row in source[["address_id", "address_text"]].itertuples(index=False)]
    pd.DataFrame(rows).to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"rows {len(rows)} written to {args.output}")


if __name__ == "__main__":
    main()
