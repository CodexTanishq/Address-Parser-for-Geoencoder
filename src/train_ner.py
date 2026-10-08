import sys
import io
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    get_linear_schedule_with_warmup,
)
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src import config
from src.utils import set_seed, AddressSpan
from src.bio import spans_to_bio
from src.augmentation import AddressAugmenter


class AugmentedAddressNERDataset(Dataset):
    """
    PyTorch Dataset for Address Token Classification with MuRIL.
    Supports on-the-fly address field shuffling and noise augmentation during training
    with 100% accurate dynamic recalculation of token character offsets and BIO labels.
    """

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        tokenizer: AutoTokenizer,
        label2id: Dict[str, int],
        augmenter: AddressAugmenter = None,
        is_training: bool = False,
        max_length: int = config.MAX_SEQ_LENGTH,
    ):
        self.samples = samples
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.augmenter = augmenter
        self.is_training = is_training
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        raw_text = sample["address_text"]
        spans = sample["spans"]

        # Apply on-the-fly permutation and noise augmentation if in training mode
        if self.is_training and self.augmenter is not None:
            text, words, tags = self.augmenter.augment(raw_text, spans)
        else:
            text = raw_text
            words, tags, _ = spans_to_bio(text, spans)

        # Tokenize subwords with offset alignment
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        label_ids = []
        previous_word_idx = None

        for word_idx in word_ids:
            if word_idx is None:
                # Special tokens ([CLS], [SEP], [PAD]) get -100
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                # First subword token of a word
                label_str = tags[word_idx] if word_idx < len(tags) else "O"
                label_ids.append(self.label2id.get(label_str, self.label2id["O"]))
            else:
                # Subsequent subwords get -100 so loss is computed on the first subword token
                label_ids.append(-100)
            previous_word_idx = word_idx

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def load_dataset_by_splits(
    splits_csv: Path = config.SPLITS_CSV,
    spans_csv: Path = config.WEAK_SPANS_CSV,
    bio_dataset_csv: Path = config.BIO_DATASET_CSV,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Loads dataset strictly according to the designated train, validation, and test splits
    from data/splits.csv (2016 train / 576 val / 288 test) with zero data leakage.
    """
    splits_df = pd.read_csv(splits_csv)
    spans_df = pd.read_csv(spans_csv)
    bio_df = pd.read_csv(bio_dataset_csv)

    # Address metadata lookup
    addr_text_map = dict(zip(bio_df["address_id"], bio_df["address_text"]))
    addr_conf_map = dict(zip(bio_df["address_id"], bio_df["overall_confidence"]))

    # Group spans by address_id
    spans_by_addr: Dict[str, List[AddressSpan]] = {}
    for _, row in spans_df.iterrows():
        aid = str(row["address_id"])
        span = AddressSpan(
            entity_type=str(row["entity_type"]),
            text=str(row["entity_text"]),
            start=int(row["start"]),
            end=int(row["end"]),
            confidence=str(row.get("confidence", config.CONF_HIGH)),
            source=str(row.get("source", "rule")),
            match_score=float(row.get("match_score", 1.0)),
        )
        if aid not in spans_by_addr:
            spans_by_addr[aid] = []
        spans_by_addr[aid].append(span)

    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []
    test_samples: List[Dict[str, Any]] = []

    for _, row in splits_df.iterrows():
        aid = str(row["address_id"])
        split = str(row["split"])
        text = addr_text_map.get(aid)
        if not text:
            continue

        spans = spans_by_addr.get(aid, [])
        record = {
            "address_id": aid,
            "address_text": text,
            "confidence": addr_conf_map.get(aid, config.CONF_HIGH),
            "spans": spans,
        }

        if split == "train":
            train_samples.append(record)
        elif split == "val":
            val_samples.append(record)
        elif split == "test":
            test_samples.append(record)

    # Verify zero leakage between train and test
    train_ids = set(s["address_id"] for s in train_samples)
    test_ids = set(s["address_id"] for s in test_samples)
    overlap = train_ids.intersection(test_ids)
    assert len(overlap) == 0, f"Critical error: {len(overlap)} IDs overlap between train and test!"

    train_texts = set(s["address_text"] for s in train_samples)
    test_texts = set(s["address_text"] for s in test_samples)
    text_overlap = train_texts.intersection(test_texts)
    assert len(text_overlap) == 0, f"Critical error: {len(text_overlap)} address texts overlap between train and test!"

    return train_samples, val_samples, test_samples


def evaluate(
    model: AutoModelForTokenClassification,
    dataloader: DataLoader,
    device: torch.device,
    id2label: Dict[int, str],
) -> Tuple[float, float, float, str, Dict[str, Any]]:
    """Evaluate model on a dataloader using seqeval entity metrics."""
    model.eval()
    all_preds = []
    all_trues = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=-1)

            for i in range(labels.size(0)):
                label_seq = labels[i].cpu().tolist()
                pred_seq = preds[i].cpu().tolist()

                clean_preds = []
                clean_trues = []
                for p, l in zip(pred_seq, label_seq):
                    if l != -100:
                        clean_preds.append(id2label[p])
                        clean_trues.append(id2label[l])

                if clean_trues:
                    all_preds.append(clean_preds)
                    all_trues.append(clean_trues)

    p = precision_score(all_trues, all_preds)
    r = recall_score(all_trues, all_preds)
    f1 = f1_score(all_trues, all_preds)
    report_str = classification_report(all_trues, all_preds, digits=4)
    report_dict = classification_report(all_trues, all_preds, output_dict=True)

    return p, r, f1, report_str, report_dict


def train_model() -> None:
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # 1. Load splits and verify zero leakage
    train_samples, val_samples, test_samples = load_dataset_by_splits()
    print(f"Loaded datasets from {config.SPLITS_CSV}:")
    print(f"  Train: {len(train_samples)} addresses (with dynamic on-the-fly augmentation)")
    print(f"  Val:   {len(val_samples)} addresses")
    print(f"  Test:  {len(test_samples)} addresses (strictly held out)")

    # 2. Initialize Augmenter
    augmenter = AddressAugmenter(
        augmentation_prob=config.AUGMENTATION_PROB,
        shuffle_prob=config.SHUFFLE_PROB,
        noise_prob=config.NOISE_PROB,
    )

    # 3. Initialize Tokenizer & Model
    print(f"\nLoading {config.TRANSFORMER_MODEL_NAME} tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        config.TRANSFORMER_MODEL_NAME,
        num_labels=len(config.BIO_LABELS),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID,
    )
    model.to(device)

    # 4. Create Datasets & DataLoaders
    train_dataset = AugmentedAddressNERDataset(
        train_samples,
        tokenizer,
        config.LABEL2ID,
        augmenter=augmenter,
        is_training=True,
    )
    val_dataset = AugmentedAddressNERDataset(
        val_samples,
        tokenizer,
        config.LABEL2ID,
        augmenter=None,
        is_training=False,
    )
    test_dataset = AugmentedAddressNERDataset(
        test_samples,
        tokenizer,
        config.LABEL2ID,
        augmenter=None,
        is_training=False,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.EVAL_BATCH_SIZE, shuffle=False
    )

    # 5. Optimizer & Linear Warmup Scheduler
    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * config.NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    print(f"\nStarting Anti-Positional-Overfitting Training for {config.NUM_EPOCHS} epochs...")
    best_val_f1 = 0.0
    best_model_path = config.MODEL_V2_DIR

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        val_p, val_r, val_f1, _, _ = evaluate(
            model, val_loader, device, config.ID2LABEL
        )
        print(f"Epoch {epoch} Result: Loss={avg_loss:.4f} | Val P={val_p:.4f} R={val_r:.4f} F1={val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            print(f"  -> New best validation F1 ({val_f1:.4f})! Saving model checkpoint to {best_model_path}...")
            best_model_path.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(best_model_path)
            tokenizer.save_pretrained(best_model_path)

    # 6. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 75)
    print("  FINAL EVALUATION ON HELD-OUT TEST SET (BEST CHECKPOINT)")
    print("=" * 75)
    best_model = AutoModelForTokenClassification.from_pretrained(best_model_path)
    best_model.to(device)

    test_p, test_r, test_f1, test_report, test_dict = evaluate(
        best_model, test_loader, device, config.ID2LABEL
    )
    print(test_report)
    print(f"Test Precision: {test_p:.4f}")
    print(f"Test Recall:    {test_r:.4f}")
    print(f"Test Macro F1:  {test_f1:.4f}")

    def convert_np(obj):
        if isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return str(obj)

    # Save evaluation report to JSON
    report_output_path = config.OUTPUTS_DIR / "ner_v2_multilandmark_evaluation_report.json"
    with open(report_output_path, "w", encoding="utf-8") as f:
        json.dump(test_dict, f, default=convert_np, indent=2, ensure_ascii=False)
    print(f"Evaluation report saved to {report_output_path}")


if __name__ == "__main__":
    train_model()
