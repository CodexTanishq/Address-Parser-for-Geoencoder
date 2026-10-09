"""Choose coordinate candidates from 08b NER output.  Validation/holdout only."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import AutoModelForCausalLM, AutoTokenizer


P = Path(__file__).resolve().parents[1]
DATA, OUT = P / "data", P / "outputs"
CHOICE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PIN_RE = re.compile(r"\b\d{5,6}\b")


def require_columns(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def landmark_spans(train: pd.DataFrame) -> list[tuple[str, str]]:
    examples = []
    for _, group in train.groupby("address_id", sort=False):
        group = group.sort_values("word_index", key=lambda x: x.astype(int))
        words, tags = group.word.tolist(), group.tag.tolist()
        current = []
        spans = []
        for word, tag in zip(words, tags):
            if tag == "B-LANDMARK":
                if current:
                    spans.append(current)
                current = [word]
            elif tag == "I-LANDMARK" and current:
                current.append(word)
            else:
                if current:
                    spans.append(current)
                current = []
        if current:
            spans.append(current)
        types = [x for x in group.landmark_type.tolist() if x and x != "temple_unknown"]
        if spans and types:
            examples.append((" ".join(spans[0]), pd.Series(types).mode().iloc[0]))
    return examples


class TypeChooser:
    """Score only the supplied landmark types with the local Qwen model."""

    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(CHOICE_MODEL, local_files_only=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            CHOICE_MODEL, local_files_only=True, torch_dtype=dtype
        ).to(self.device).eval()

    def log_likelihoods(self, span: str, type_names: list[str]) -> torch.Tensor:
        instruction = (
            "This Indian address landmark may be written in Hindi or Kannada in English letters. "
            f"Which type is it? Landmark: {span}. Type:\n"
            f"Allowed types: {', '.join(type_names)}\nType:"
        )
        messages = [{"role": "user", "content": instruction}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prefix_ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt").input_ids[0]
        scores: list[torch.Tensor] = []
        for name in type_names:
            choice_ids = self.tokenizer(" " + name, add_special_tokens=False, return_tensors="pt").input_ids[0]
            ids = torch.cat([prefix_ids, choice_ids]).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids).logits[0]
            start = len(prefix_ids)
            token_log_probs = torch.log_softmax(logits[start - 1:-1], dim=-1)
            scores.append(token_log_probs.gather(1, choice_ids.to(self.device).unsqueeze(1)).sum())
        return torch.stack(scores).detach().cpu()

def type_from_avg_llm(
    span: str,
    old_type: str,
    probability: float,
    type_names: list[str],
    classifier_distribution: torch.Tensor,
    chooser: TypeChooser | None,
    content_free_scores: torch.Tensor | None,
) -> tuple[str, str, float | str, TypeChooser | None, torch.Tensor | None]:
    if probability >= 0.5:
        return old_type, "classifier", "", chooser, content_free_scores
    if chooser is None:
        chooser = TypeChooser()
    if content_free_scores is None:
        content_free_scores = chooser.log_likelihoods("N/A", type_names)
    qwen_distribution = torch.softmax(chooser.log_likelihoods(span, type_names) - content_free_scores, dim=0)
    average = (classifier_distribution + qwen_distribution) / 2
    top = int(torch.argmax(average))
    return type_names[top], "avg_llm", float(average[top]), chooser, content_free_scores


def nearest_locality_catchments(pois: pd.DataFrame, localities: pd.DataFrame) -> pd.DataFrame:
    assigned = []
    for row in pois.itertuples(index=False):
        choices = localities[localities.town_id.eq(row.town_id)]
        if choices.empty:
            raise ValueError(f"No locality is available in POI town {row.town_id}")
        dist2 = (choices.centroid_x - float(row.x)) ** 2 + (choices.centroid_y - float(row.y)) ** 2
        assigned.append(choices.loc[dist2.idxmin(), "locality_id"])
    result = pois.copy()
    result["catchment_locality_id"] = assigned
    return result


def pick_candidates(
    poi: pd.DataFrame,
    localities: pd.DataFrame,
    town_id: str,
    locality_id: str,
    pincode: str,
    landmark_type: str,
) -> tuple[pd.DataFrame, int]:
    base = poi[(poi.town_id.eq(town_id)) & (poi.landmark_type.eq(landmark_type))]
    if base.empty:
        return base, 4
    if locality_id:
        at_locality = base[base.catchment_locality_id.eq(locality_id)]
        if not at_locality.empty:
            return at_locality, 1
        loc_row = localities[localities.locality_id.eq(locality_id)]
        if not loc_row.empty:
            same_pin = localities[(localities.town_id.eq(town_id)) & (localities.pincode.eq(loc_row.iloc[0].pincode))]
            level_two = base[base.catchment_locality_id.isin(same_pin.locality_id)]
            if not level_two.empty:
                return level_two, 2
    if not locality_id and pincode:
        pin_locs = localities[(localities.town_id.eq(town_id)) & (localities.pincode.eq(pincode))]
        level_three = base[base.catchment_locality_id.isin(pin_locs.locality_id)]
        if not level_three.empty:
            return level_three, 3
    return base, 4


def run(input_path: Path, output_prefix: Path) -> None:
    ner = pd.read_csv(input_path, dtype=str).fillna("")
    baseline = pd.read_csv(DATA / "baseline_geocodes.csv", dtype=str).fillna("")
    train = pd.read_csv(DATA / "labels" / "train_bio_merged_v2_noout.csv", dtype=str).fillna("")
    loc = pd.read_csv(DATA / "localities.csv", dtype=str).fillna("")
    poi = pd.read_csv(DATA / "landmarks_poi.csv", dtype=str).fillna("")
    for name, frame, columns in [
        ("NER input", ner, ["address_id", "raw_text", "town_id", "locality_id"]),
        ("baseline_geocodes", baseline, ["address_id", "geocoder_x", "geocoder_y", "precision"]),
        ("train", train, ["address_id", "word_index", "word", "tag", "landmark_type"]),
        ("localities", loc, ["locality_id", "town_id", "pincode", "centroid_x", "centroid_y"]),
        ("pois", poi, ["poi_id", "town_id", "landmark_type", "name", "x", "y"]),
    ]:
        print(name, list(frame.columns))
        require_columns(frame, columns, name)
    for column in ["centroid_x", "centroid_y"]:
        loc[column] = pd.to_numeric(loc[column], errors="raise")
    for column in ["x", "y"]:
        poi[column] = pd.to_numeric(poi[column], errors="raise")
    for column in ["geocoder_x", "geocoder_y"]:
        baseline[column] = pd.to_numeric(baseline[column], errors="raise")
    unknown_ids = sorted(set(ner.address_id) - set(baseline.address_id))
    print(f"{input_path.name}: address_ids absent from baseline_geocodes.csv = {len(unknown_ids)}")
    if unknown_ids:
        print("missing IDs", unknown_ids[:20])
    joined = ner.merge(baseline, on="address_id", how="left", validate="one_to_one")
    if joined.geocoder_x.isna().any():
        raise ValueError("Cannot choose final POIs: some input addresses have no baseline geocode")

    examples = landmark_spans(train)
    if not examples:
        raise ValueError("No labelled train landmark spans available for classifier")
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 5))
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    classifier.fit(vectorizer.fit_transform([x[0] for x in examples]), [x[1] for x in examples])
    types = sorted(poi.landmark_type.unique())
    if set(classifier.classes_) - set(types):
        raise ValueError("Classifier has landmark types not present in landmarks_poi.csv")
    poi = nearest_locality_catchments(poi, loc)
    town_centres = loc.groupby("town_id", as_index=False)[["centroid_x", "centroid_y"]].mean()
    chooser: TypeChooser | None = None
    content_free_scores: torch.Tensor | None = None
    result, low_rows = [], []
    for row in joined.itertuples(index=False):
        town_id, locality_id = row.town_id, row.locality_id
        pin = next(iter(PIN_RE.findall(row.raw_text)), "")
        locality_row = loc[loc.locality_id.eq(locality_id)] if locality_id else loc.iloc[0:0]
        if not town_id:
            # No town means neither a pincode area nor a town centre is safe.
            area_level, area_x, area_y = "unknown_town", "", ""
        elif not locality_row.empty:
            area_level, area_x, area_y = "locality", locality_row.iloc[0].centroid_x, locality_row.iloc[0].centroid_y
        elif pin:
            pin_locs = loc[(loc.town_id.eq(town_id)) & (loc.pincode.eq(pin))]
            if not pin_locs.empty:
                area_level, area_x, area_y = "pincode_area", pin_locs.centroid_x.mean(), pin_locs.centroid_y.mean()
            else:
                pin_locs = loc.iloc[0:0]
                centre = town_centres[town_centres.town_id.eq(town_id)]
                area_level, area_x, area_y = "town", centre.iloc[0].centroid_x, centre.iloc[0].centroid_y
        else:
            pin_locs = loc.iloc[0:0]
            centre = town_centres[town_centres.town_id.eq(town_id)]
            area_level, area_x, area_y = "town", centre.iloc[0].centroid_x, centre.iloc[0].centroid_y
        out = {"address_id": row.address_id, "raw_text": row.raw_text, "town_id": town_id,
               "locality_id": locality_id, "area_level": area_level, "area_x": area_x, "area_y": area_y}
        for number in (1, 2):
            span = getattr(row, f"lm{number}_span", "") if f"lm{number}_span" in joined.columns else ""
            probability_text = getattr(row, f"lm{number}_prob", "") if f"lm{number}_prob" in joined.columns else ""
            old_type = getattr(row, f"lm{number}_type", "") if f"lm{number}_type" in joined.columns else ""
            prefix = f"lm{number}_"
            out.update({prefix + k: "" for k in ["span", "prob", "type", "type_source", "level", "n_candidates", "poi_id", "x", "y", "dist_to_geocode", "geocode_precision"]})
            if not span or not old_type:
                continue
            probability = float(probability_text)
            raw_probabilities = classifier.predict_proba(vectorizer.transform([span]))[0]
            classifier_distribution = torch.zeros(len(types), dtype=torch.float32)
            for label, value in zip(classifier.classes_, raw_probabilities):
                classifier_distribution[types.index(label)] = float(value)
            final_type, source, avg_top_prob, chooser, content_free_scores = type_from_avg_llm(
                span, old_type, probability, types, classifier_distribution, chooser, content_free_scores
            )
            if probability < 0.5:
                low_rows.append({"span": span, "classifier_type": old_type, "new_type": final_type,
                                 "avg_top_prob": avg_top_prob, "type_source": source})
            if not town_id:
                raise ValueError(f"Cannot choose a town-restricted POI for {row.address_id}: predicted town is blank")
            candidates, level = pick_candidates(poi, loc, town_id, locality_id, pin, final_type)
            if candidates.empty:
                raise ValueError(f"No candidate POI for address {row.address_id}, type {final_type}, town {town_id}")
            dist = ((candidates.x - row.geocoder_x) ** 2 + (candidates.y - row.geocoder_y) ** 2) ** 0.5
            chosen = candidates.loc[dist.idxmin()]
            out.update({prefix + "span": span, prefix + "prob": probability, prefix + "type": final_type,
                        prefix + "type_source": source, prefix + "level": level, prefix + "n_candidates": len(candidates),
                        prefix + "poi_id": chosen.poi_id, prefix + "x": chosen.x, prefix + "y": chosen.y,
                        prefix + "dist_to_geocode": float(dist.loc[dist.idxmin()]), prefix + "geocode_precision": row.precision})
        result.append(out)
    out_frame = pd.DataFrame(result)
    low_frame = pd.DataFrame(low_rows, columns=["span", "classifier_type", "new_type", "avg_top_prob", "type_source"])
    output_csv = output_prefix.with_suffix(".csv")
    low_csv = output_prefix.with_name(output_prefix.name + "_low_probability_spans").with_suffix(".csv")
    report = output_prefix.with_name(output_prefix.name + "_report").with_suffix(".txt")
    out_frame.to_csv(output_csv, index=False, encoding="utf-8-sig")
    low_frame.to_csv(low_csv, index=False, encoding="utf-8-sig")
    levels = pd.concat([out_frame.lm1_level, out_frame.lm2_level]).replace("", pd.NA).dropna().value_counts().to_dict()
    nc = pd.to_numeric(pd.concat([out_frame.lm1_n_candidates, out_frame.lm2_n_candidates]).replace("", pd.NA).dropna())
    spread = {"1": int((nc == 1).sum()), "2-3": int(nc.between(2, 3).sum()), "4-10": int(nc.between(4, 10).sum()), ">10": int((nc > 10).sum())}
    changed = int((low_frame.classifier_type != low_frame.new_type).sum()) if not low_frame.empty else 0
    report_text = "\n".join([
        f"input {input_path.name}", f"rows {len(out_frame)}", f"absent baseline IDs {len(unknown_ids)}",
        "low-probability model rule: equal average of classifier probabilities and content-free-calibrated Qwen/Qwen2.5-0.5B-Instruct probabilities",
        f"rows per area_level {out_frame.area_level.value_counts().to_dict()}", f"rows per candidate level {levels}",
        f"types changed by avg rule {changed}",
        f"rows by geocoder precision {baseline[baseline.address_id.isin(ner.address_id)].precision.value_counts().to_dict()}",
        f"n_candidates spread {spread}", f"zero-candidate landmark rows 0", "coordinates cannot be scored: no true coordinates",
    ]) + "\n"
    report.write_text(report_text, encoding="utf-8")
    print(report_text)
    print("low probability spans table", low_csv)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_prefix", type=Path)
    args = parser.parse_args()
    run(args.input_csv, args.output_prefix)


if __name__ == "__main__":
    main()
