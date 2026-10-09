"""Fine-tune MuRIL on merged train BIO labels and evaluate on validation only."""

from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import classification_report, f1_score
from transformers import AutoModelForTokenClassification, AutoTokenizer, DataCollatorForTokenClassification, Trainer, TrainingArguments, set_seed


PROJECT = Path(__file__).resolve().parents[1]
TRAIN_FILE = PROJECT / "data" / "labels" / "train_bio_merged.csv"
VALIDATION_FILE = PROJECT / "data" / "labels" / "validation_bio.csv"
MODEL_NAME = "google/muril-base-cased"
MODEL_DIR = PROJECT / "models" / "ner_v1"
WORK_DIR = PROJECT / "temp_code" / "ner_training"
LABELS = ["O", "B-TOWN", "I-TOWN", "B-LANDMARK", "I-LANDMARK", "B-LOCALITY", "I-LOCALITY", "B-RELATION", "I-RELATION", "B-PINCODE"]
LABEL_TO_ID = {label: index for index, label in enumerate(LABELS)}


def typo(word: str, rng: random.Random) -> str:
    if len(word) < 4 or word.isdigit():
        return word
    index = rng.randrange(1, len(word))
    return word[:index - 1] + word[index] + word[index - 1] + word[index + 1:]


def addresses_from_bio(path: Path, noisy: bool = False) -> list[tuple[list[str], list[str]]]:
    table = pd.read_csv(path, dtype=str).fillna("")
    required = {"address_id", "word_index", "word", "tag"}
    if not required.issubset(table.columns):
        raise ValueError(f"{path} is missing required columns")
    rng = random.Random(42)
    rows = []
    for _, group in table.groupby("address_id", sort=False):
        group = group.sort_values("word_index", key=lambda values: values.astype(int))
        words, tags = group["word"].tolist(), group["tag"].tolist()
        if noisy:
            words = [typo(word, rng) if rng.random() < 0.10 else word for word in words]
        rows.append((words, tags))
    return rows


class BioDataset(torch.utils.data.Dataset):
    def __init__(self, rows: list[tuple[list[str], list[str]]], tokenizer):
        self.rows = rows
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        words, tags = self.rows[index]
        encoding = self.tokenizer(words, is_split_into_words=True, truncation=True, max_length=128)
        label_ids, previous_word = [], None
        for word_id in encoding.word_ids():
            if word_id is None or word_id == previous_word:
                label_ids.append(-100)
            elif tags[word_id] == "B-PINCODE":
                label_ids.append(-100)  # Pincode is a rule, not NER loss.
            else:
                label_ids.append(LABEL_TO_ID[tags[word_id]])
            previous_word = word_id
        encoding["labels"] = label_ids
        return {key: torch.tensor(value) for key, value in encoding.items()}


def metric_fn(prediction) -> dict[str, float]:
    logits, labels = prediction
    predictions = np.argmax(logits, axis=-1)
    true_tags, predicted_tags = [], []
    for predicted, actual in zip(predictions, labels):
        keep = actual != -100
        true_tags.append([LABELS[label] for label in actual[keep]])
        predicted_tags.append([LABELS[label] for label in predicted[keep]])
    return {"f1": f1_score(true_tags, predicted_tags)}


def report_per_type(trainer: Trainer, dataset: BioDataset, title: str) -> None:
    output = trainer.predict(dataset)
    predictions = np.argmax(output.predictions, axis=-1)
    true_tags, predicted_tags = [], []
    for predicted, actual in zip(predictions, output.label_ids):
        keep = actual != -100
        true_tags.append([LABELS[label] for label in actual[keep]])
        predicted_tags.append([LABELS[label] for label in predicted[keep]])
    print(title)
    print(classification_report(true_tags, predicted_tags, digits=4, zero_division=0))


def make_args(batch_size: int, accumulation: int) -> TrainingArguments:
    return TrainingArguments(
        output_dir=str(WORK_DIR), learning_rate=3e-5, per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size, gradient_accumulation_steps=accumulation,
        num_train_epochs=5, fp16=True, seed=42, eval_strategy="epoch", save_strategy="epoch",
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        report_to=[], logging_steps=25,
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Install CUDA-enabled PyTorch before training this GPU job.")
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_NAME, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id=LABEL_TO_ID)
    train_data = BioDataset(addresses_from_bio(TRAIN_FILE, noisy=True), tokenizer)
    validation_data = BioDataset(addresses_from_bio(VALIDATION_FILE), tokenizer)
    noisy_validation_data = BioDataset(addresses_from_bio(VALIDATION_FILE, noisy=True), tokenizer)
    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(model=model, args=make_args(16, 1), train_dataset=train_data, eval_dataset=validation_data, data_collator=collator, compute_metrics=metric_fn)
    try:
        trainer.train()
    except RuntimeError as error:
        if "out of memory" not in str(error).lower():
            raise
        print("CUDA out of memory: retrying with batch size 8 and gradient accumulation 2.")
        torch.cuda.empty_cache()
        trainer = Trainer(model=model, args=make_args(8, 2), train_dataset=train_data, eval_dataset=validation_data, data_collator=collator, compute_metrics=metric_fn)
        trainer.train()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    report_per_type(trainer, validation_data, "Validation per-type F1")
    report_per_type(trainer, noisy_validation_data, "Noisy validation per-type F1")
    print("LANDMARK and RELATION validation labels come only from 02_auto_label.py, so their F1 can look low.")


if __name__ == "__main__":
    main()
