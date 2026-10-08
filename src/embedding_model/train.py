import sys
import io
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm

from src.embedding_model import config
from src.embedding_model.model import HierarchicalAddressEncoder
from src.embedding_model.preprocessing import preprocess_address
from src.embedding_model.losses import info_nce_loss, triplet_margin_loss, hierarchy_compatibility_loss
from src.embedding_model.hierarchy import GazetteerIndex
from src.embedding_model.dataset import prepare_embedding_datasets
from src.utils import set_seed, load_raw_data


class ContrastiveAddressDataset(Dataset):
    """
    Constructs training batches with positive targets, contextual entity strings,
    and dynamically sampled hard negatives for Town, Locality, and Landmark.
    """
    def __init__(
        self,
        df: pd.DataFrame,
        gazetteer: GazetteerIndex,
        preprocessing_mode: str = "raw",
    ):
        self.records = []
        self.gazetteer = gazetteer
        self.preprocessing_mode = preprocessing_mode

        for _, r in df.iterrows():
            raw_addr = str(r["address_text"])
            addr_text = preprocess_address(raw_addr, mode=preprocessing_mode)
            tid = str(r["town_id"])
            tname = str(r["town_name"]) if pd.notnull(r.get("town_name")) else gazetteer.towns.get(tid, "")
            lname = str(r["locality_name"]) if pd.notnull(r.get("locality_name")) and str(r.get("locality_name")) != "" else "UNKNOWN"
            landmarks = str(r["landmarks"]) if pd.notnull(r.get("landmarks")) and str(r.get("landmarks")) != "" else "UNKNOWN"
            # Extract first landmark if list
            primary_lm = landmarks.split("|")[0].strip() if "|" in landmarks else landmarks.strip()

            self.records.append({
                "address_id": str(r["address_id"]),
                "address_text": addr_text,
                "town_id": tid,
                "town_name": tname,
                "locality_name": lname,
                "landmark_name": primary_lm,
            })

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.records[idx]
        tid = item["town_id"]
        tname = item["town_name"]
        lname = item["locality_name"]
        lm = item["landmark_name"]

        # Positive strings
        pos_town_str = tname
        pos_loc_str = f"{lname}, {tname}" if lname != "UNKNOWN" else tname
        pos_lm_str = f"{lm}, {lname}, {tname}" if lm != "UNKNOWN" else pos_loc_str

        # Hard negative generation
        neg_town_name = self.gazetteer.get_hard_negative_town(tid)
        neg_loc_name, neg_loc_town = self.gazetteer.get_hard_negative_locality(correct_loc_id="UNKNOWN", correct_town_id=tid)
        neg_loc_str = f"{neg_loc_name}, {neg_loc_town}"
        neg_lm_name = self.gazetteer.get_hard_negative_landmark(lm, loc_id="UNKNOWN")
        neg_lm_str = f"{neg_lm_name}, {lname}, {tname}"

        return {
            "address_text": item["address_text"],
            "pos_town_str": pos_town_str,
            "neg_town_str": neg_town_name,
            "pos_loc_str": pos_loc_str,
            "neg_loc_str": neg_loc_str,
            "pos_lm_str": pos_lm_str,
            "neg_lm_str": neg_lm_str,
            "has_locality": (lname != "UNKNOWN"),
            "has_landmark": (lm != "UNKNOWN"),
        }


def collate_fn(batch: List[Dict[str, Any]], tokenizer) -> Dict[str, Any]:
    def _encode(texts: List[str]):
        return tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        )

    return {
        "address": _encode([b["address_text"] for b in batch]),
        "pos_town": _encode([b["pos_town_str"] for b in batch]),
        "neg_town": _encode([b["neg_town_str"] for b in batch]),
        "pos_loc": _encode([b["pos_loc_str"] for b in batch]),
        "neg_loc": _encode([b["neg_loc_str"] for b in batch]),
        "pos_lm": _encode([b["pos_lm_str"] for b in batch]),
        "neg_lm": _encode([b["neg_lm_str"] for b in batch]),
        "has_locality": torch.tensor([b["has_locality"] for b in batch], dtype=torch.bool),
        "has_landmark": torch.tensor([b["has_landmark"] for b in batch], dtype=torch.bool),
    }


def train_hierarchical_model(
    pooling_mode: str = "attention",
    loss_type: str = "infonce",
    use_hard_negatives: bool = True,
    use_hierarchy: bool = True,
    preprocessing_mode: str = "raw",
    num_epochs: int = config.NUM_EPOCHS,
    save_model: bool = True,
) -> Tuple[HierarchicalAddressEncoder, AutoTokenizer, GazetteerIndex, Dict[str, float]]:
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load gazetteers
    addresses_df, towns_df, localities_df = load_raw_data()
    gazetteer = GazetteerIndex(towns_df, localities_df)

    # Load splits
    train_df, val_df, test_df, _ = prepare_embedding_datasets()

    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    model = HierarchicalAddressEncoder(
        model_name=config.PRETRAINED_MODEL_NAME,
        projection_dim=config.PROJECTION_DIM,
        pooling_mode=pooling_mode,
    ).to(device)

    train_dataset = ContrastiveAddressDataset(train_df, gazetteer, preprocessing_mode=preprocessing_mode)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer)
    )

    optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    print(f"\n[Training] Pooling={pooling_mode} | Loss={loss_type} | Hierarchy={use_hierarchy} | HardNegs={use_hard_negatives} | Preprocessing={preprocessing_mode}")
    history = {"epoch_losses": []}

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}")
        for batch in pbar:
            optimizer.zero_grad()

            # Encode Address
            addr_inputs = batch["address"].to(device)
            addr_emb = model.encode_address(addr_inputs["input_ids"], addr_inputs["attention_mask"])  # [B, D]

            # Encode Town Pos/Neg
            pos_t_inputs = batch["pos_town"].to(device)
            pos_t_emb = model.encode_entity(pos_t_inputs["input_ids"], pos_t_inputs["attention_mask"])

            if use_hard_negatives:
                neg_t_inputs = batch["neg_town"].to(device)
                neg_t_emb = model.encode_entity(neg_t_inputs["input_ids"], neg_t_inputs["attention_mask"])
            else:
                neg_t_emb = torch.roll(pos_t_emb, shifts=1, dims=0)

            # Compute Town Loss
            if loss_type == "infonce":
                l_town = info_nce_loss(addr_emb, pos_t_emb, neg_t_emb)
            else:
                l_town = triplet_margin_loss(addr_emb, pos_t_emb, neg_t_emb)

            # Encode Locality Pos/Neg
            pos_l_inputs = batch["pos_loc"].to(device)
            pos_l_emb = model.encode_entity(pos_l_inputs["input_ids"], pos_l_inputs["attention_mask"])

            if use_hard_negatives:
                neg_l_inputs = batch["neg_loc"].to(device)
                neg_l_emb = model.encode_entity(neg_l_inputs["input_ids"], neg_l_inputs["attention_mask"])
            else:
                neg_l_emb = torch.roll(pos_l_emb, shifts=1, dims=0)

            if loss_type == "infonce":
                l_loc = info_nce_loss(addr_emb, pos_l_emb, neg_l_emb)
            else:
                l_loc = triplet_margin_loss(addr_emb, pos_l_emb, neg_l_emb)

            # Encode Landmark Pos/Neg
            pos_m_inputs = batch["pos_lm"].to(device)
            pos_m_emb = model.encode_entity(pos_m_inputs["input_ids"], pos_m_inputs["attention_mask"])

            if use_hard_negatives:
                neg_m_inputs = batch["neg_lm"].to(device)
                neg_m_emb = model.encode_entity(neg_m_inputs["input_ids"], neg_m_inputs["attention_mask"])
            else:
                neg_m_emb = torch.roll(pos_m_emb, shifts=1, dims=0)

            if loss_type == "infonce":
                l_lm = info_nce_loss(addr_emb, pos_m_emb, neg_m_emb)
            else:
                l_lm = triplet_margin_loss(addr_emb, pos_m_emb, neg_m_emb)

            # Hierarchy Compatibility Loss
            if use_hierarchy:
                l_hier = hierarchy_compatibility_loss(pos_l_emb, pos_t_emb)
            else:
                l_hier = torch.tensor(0.0, device=device)

            # Combined Objective
            batch_loss = (
                config.LAMBDA_TOWN * l_town
                + config.LAMBDA_LOCALITY * l_loc
                + config.LAMBDA_LANDMARK * l_lm
                + config.LAMBDA_HIERARCHY * l_hier
            )

            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += batch_loss.item()
            pbar.set_postfix({"loss": f"{batch_loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)
        history["epoch_losses"].append(avg_loss)
        print(f"Epoch {epoch} Completed | Loss: {avg_loss:.4f}")

    if save_model:
        save_path = config.MODELS_DIR
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path / "model_state.pt")
        tokenizer.save_pretrained(save_path)
        print(f"[Model Saved] State saved to {save_path / 'model_state.pt'}")

    return model, tokenizer, gazetteer, history
