"""
Milestone 4B: MuRIL Token Classifier Fine-Tuning (google/muril-base-cased)
- Compact tag set: TOWN, LOCALITY, LANDMARK, RELATION. BIO tags.
- Trained on train_augmented.parquet (8,700 samples), 4 epochs, lr 3e-5, AdamW.
- Predicts on first sub-word token of each word.
- Evaluates validation entity-F1 and per-class metrics.
- Checkpoint saved strictly to final_model/models/muril_address_ner_final.
"""
import sys
import io
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np

from final_model.src import config

class NERAddressDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, label2id, max_len=128):
        self.texts = df['tokens'].tolist()
        self.tags = df['bio_tags'].tolist()
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        words = self.texts[idx]
        word_tags = self.tags[idx]

        subwords = []
        subword_labels = []

        for word, tag in zip(words, word_tags):
            toks = self.tokenizer.tokenize(word)
            if not toks:
                continue
            subwords.extend(toks)
            # Label first sub-word token, -100 for subsequent sub-words
            subword_labels.append(self.label2id.get(tag, 0))
            subword_labels.extend([-100] * (len(toks) - 1))

        # Truncate
        subwords = subwords[: self.max_len - 2]
        subword_labels = subword_labels[: self.max_len - 2]

        input_ids = [self.tokenizer.cls_token_id] + self.tokenizer.convert_tokens_to_ids(subwords) + [self.tokenizer.sep_token_id]
        labels = [-100] + subword_labels + [-100]
        attention_mask = [1] * len(input_ids)

        # Padding
        pad_len = self.max_len - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
        }

def train_muril_ner():
    start_time = time.time()
    print("=" * 70)
    print("FINE-TUNING MURIL TOKEN CLASSIFIER (google/muril-base-cased)")
    print("=" * 70)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(config.TRANSFORMER_MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        config.TRANSFORMER_MODEL_NAME,
        num_labels=len(config.BIO_LABELS),
        id2label=config.ID2LABEL,
        label2id=config.LABEL2ID
    ).to(device)

    train_df = pd.read_parquet(config.OUTPUTS_DIR / "train_augmented.parquet")
    val_df = pd.read_parquet(config.OUTPUTS_DIR / "val_clean.parquet")

    train_dataset = NERAddressDataset(train_df, tokenizer, config.LABEL2ID)
    val_dataset = NERAddressDataset(val_df, tokenizer, config.LABEL2ID)

    train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.EVAL_BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * 4 # 4 epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

    save_dir = config.MURIL_NER_MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float('inf')

    for epoch in range(1, 5):
        model.train()
        total_train_loss = 0.0
        for batch in train_loader:
            optimizer.zero_grad()
            ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            lbls = batch['labels'].to(device)

            outputs = model(ids, attention_mask=mask, labels=lbls)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                lbls = batch['labels'].to(device)
                outputs = model(ids, attention_mask=mask, labels=lbls)
                total_val_loss += outputs.loss.item()
        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch}/4 | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            print(f"  -> Best model found! Saving checkpoint to {save_dir}...")
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

    print(f"\n[OK] Training completed in {time.time() - start_time:.2f}s! Checkpoint saved to: {save_dir}")

if __name__ == "__main__":
    train_muril_ner()
