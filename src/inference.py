import sys
import io
import json
import re
from pathlib import Path
from typing import Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src import config
from src.bio import tokenize_with_spans


class AddressNERPredictor:
    """End-to-end inference engine for structured address extraction using fine-tuned MuRIL."""

    def __init__(self, model_dir: Path = None):
        target_dir = Path(model_dir) if model_dir else config.DEFAULT_MODEL_DIR
        if not target_dir.exists():
            # Fallback to model v1 backup if default path is missing
            if config.MODEL_V1_DIR.exists():
                target_dir = config.MODEL_V1_DIR
            else:
                raise FileNotFoundError(
                    f"Trained model not found at {target_dir}. Please train the model first with: python -m src.train_ner"
                )
        self.model_dir = target_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(target_dir)
        self.model = AutoModelForTokenClassification.from_pretrained(target_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, raw_address: str) -> Dict[str, Any]:
        """
        Extract structured address entities from raw messy address text.
        Returns a dictionary containing raw address, tokens, BIO tags, and structured entities.
        """
        token_tuples = tokenize_with_spans(raw_address)
        words = [t[0] for t in token_tuples]

        if not words:
            return {
                "raw_address": raw_address,
                "structured_address": {},
                "tokens": [],
                "bio_tags": [],
                "formatted_sentence": "",
            }

        # Tokenize with subword mapping
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=config.MAX_SEQ_LENGTH,
            padding=False,
            truncation=True,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)
        word_ids = encoding.word_ids(batch_index=0)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            pred_ids = torch.argmax(logits, dim=-1)[0].cpu().tolist()

        # Map subword predictions back to words
        word_labels = ["O"] * len(words)
        seen_words = set()

        for subword_idx, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx not in seen_words:
                pred_label = config.ID2LABEL[pred_ids[subword_idx]]
                word_labels[word_idx] = pred_label
                seen_words.add(word_idx)

        # Assemble entities from BIO tags
        entities: Dict[str, List[str]] = {entity: [] for entity in config.ENTITIES}
        curr_entity = None
        curr_tokens = []

        for word, tag in zip(words, word_labels):
            if tag.startswith("B-"):
                if curr_entity and curr_tokens:
                    entities[curr_entity].append(" ".join(curr_tokens))
                curr_entity = tag[2:]
                curr_tokens = [word]
            elif tag.startswith("I-"):
                ent_type = tag[2:]
                if curr_entity == ent_type:
                    curr_tokens.append(word)
                else:
                    if curr_entity and curr_tokens:
                        entities[curr_entity].append(" ".join(curr_tokens))
                    curr_entity = ent_type
                    curr_tokens = [word]
            else:
                if curr_entity and curr_tokens:
                    entities[curr_entity].append(" ".join(curr_tokens))
                curr_entity = None
                curr_tokens = []

        # Multi-Landmark Fusion: In Model v2, recover any secondary directional/spatial landmarks
        # (e.g. 'Opp. Post Office', 'Behind SBI Bank') that might have been unassigned by BIO tagging
        # without overriding verified structured entities
        landmarks_list = list(entities["LANDMARK"])
        if "v2" in str(self.model_dir):
            from src.landmark import extract_landmarks
            # Protected spans are already assigned non-landmark entities
            assigned_non_lms = (
                entities["HOUSE_NUMBER"]
                + entities["STREET"]
                + entities["LOCALITY"]
                + entities["TOWN"]
                + entities["PINCODE"]
            )
            rule_lms = extract_landmarks(raw_address)
            for r_lm in rule_lms:
                lm_text = r_lm.text.strip(" ,.-")
                # Ensure it doesn't overlap with non-landmark entities and isn't already in list
                if not any(lm_text.lower() == existing.lower() for existing in landmarks_list):
                    if not any(lm_text.lower() in non_lm.lower() for non_lm in assigned_non_lms):
                        landmarks_list.append(lm_text)

            # Filter out over-captured composite landmarks:
            # 1. Remove candidates containing a 6-digit PINCODE
            # 2. Remove candidates longer than 35 chars that contain another smaller extracted landmark inside
            filtered_landmarks = []
            for cand in landmarks_list:
                cand_clean = cand.strip(" ,.-")
                # Filter 2a: Contains 6-digit pincode
                if re.search(r"\b\d{6}\b", cand_clean):
                    continue
                filtered_landmarks.append(cand_clean)

            # Filter 2b: Remove overly long (>35 chars) composite spans if a tighter extracted landmark exists within it
            final_landmarks = []
            for cand in filtered_landmarks:
                if len(cand) > 35:
                    has_smaller_sublandmark = any(
                        other.lower() in cand.lower() and other.lower() != cand.lower()
                        for other in filtered_landmarks
                    )
                    if has_smaller_sublandmark:
                        continue
                final_landmarks.append(cand)

            landmarks_list = final_landmarks

        # Flatten singletons into values where appropriate, keeping multi-landmarks and multi-streets as lists
        structured_output = {
            "house_number": entities["HOUSE_NUMBER"][0] if entities["HOUSE_NUMBER"] else None,
            "streets": entities["STREET"],
            "ward": entities["WARD"][0] if entities["WARD"] else None,
            "landmarks": landmarks_list,
            "locality": entities["LOCALITY"][0] if entities["LOCALITY"] else None,
            "town": entities["TOWN"][0] if entities["TOWN"] else None,
            "pincode": entities["PINCODE"][0] if entities["PINCODE"] else None,
        }

        formatted_sentence = format_address_to_english(structured_output)

        return {
            "raw_address": raw_address,
            "structured_address": structured_output,
            "tokens": words,
            "bio_tags": word_labels,
            "formatted_sentence": formatted_sentence,
        }


def format_address_to_english(structured_json: Dict[str, Any]) -> str:
    """
    Convert the parsed entity dictionary into a clear, natural English navigational statement.
    Example Input JSON:
      {
        "house_number": "12",
        "streets": ["4th Cross"],
        "locality": "Kuvempu Layout",
        "town": "Bengaluru",
        "pincode": "560001",
        "landmarks": ["Opp. Post Office", "Near Water Tank"]
      }
    Example Output Sentence:
      "Located at House No. 12, 4th Cross in Kuvempu Layout, Bengaluru (Pincode: 560001). Key landmarks: Opp. Post Office, Near Water Tank."
    """
    parts = []

    # 1. House Number & Streets
    hno = structured_json.get("house_number")
    streets = structured_json.get("streets", [])
    if isinstance(streets, str):
        streets = [streets] if streets else []

    premise_parts = []
    if hno:
        clean_hno = str(hno).strip()
        if not re.match(r"(?i)^(no\.?|house\s+no\.?|#)", clean_hno):
            premise_parts.append(f"House No. {clean_hno}")
        else:
            premise_parts.append(clean_hno)
    if streets:
        premise_parts.append(", ".join(streets))

    if premise_parts:
        parts.append(f"Located at {', '.join(premise_parts)}")
    else:
        parts.append("Located")

    # 2. Locality & Ward
    loc = structured_json.get("locality")
    ward = structured_json.get("ward")
    area_parts = []
    if ward:
        area_parts.append(f"Ward {ward}")
    if loc:
        area_parts.append(str(loc))

    if area_parts:
        parts.append(f"in {', '.join(area_parts)}")

    # 3. Town & Pincode
    town = structured_json.get("town")
    pincode = structured_json.get("pincode")

    town_pin_part = ""
    if town and pincode:
        town_pin_part = f"{town} (Pincode: {pincode})"
    elif town:
        town_pin_part = f"{town}"
    elif pincode:
        town_pin_part = f"(Pincode: {pincode})"

    main_statement = " ".join(parts).strip()
    if town_pin_part:
        if parts:
            main_statement = f"{main_statement}, {town_pin_part}."
        else:
            main_statement = f"Located in {town_pin_part}."
    else:
        main_statement = f"{main_statement}."

    # 4. Key landmarks
    landmarks = structured_json.get("landmarks", [])
    if isinstance(landmarks, str):
        landmarks = [landmarks] if landmarks else []

    if landmarks:
        lm_str = ", ".join(landmarks)
        sentence = f"{main_statement} Key landmarks: {lm_str}."
    else:
        sentence = main_statement

    return sentence


def main():
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        # Default representative demo examples
        query = "6th Cross, 5th Main, near church, opp water tank, Kuvempu Layt, Kaveripura - 960102"

    predictor = AddressNERPredictor()
    result = predictor.predict(query)

    print("=" * 65)
    print("  MuRIL ADDRESS NER INFERENCE RESULT")
    print("=" * 65)
    print(f"Input Address: {result['raw_address']}\n")
    print("Structured Extraction:")
    for k, v in result["structured_address"].items():
        print(f"  {k:15s}: {v}")
    print(f"\nNatural English Statement:\n  {result['formatted_sentence']}\n")
    print("Token BIO Tags:")
    for tok, tag in zip(result["tokens"], result["bio_tags"]):
        print(f"  {tok:20s} -> {tag}")
    print("=" * 65)


if __name__ == "__main__":
    main()
