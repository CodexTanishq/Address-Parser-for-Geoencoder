import sys
import io
from pathlib import Path
from typing import Dict, Any, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from tqdm import tqdm

if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

from src import config
from src.utils import set_seed
from src.augmentation import AddressAugmenter
from src.embedding_dataset import (
    load_spatial_training_splits,
    load_canonical_spatial_database,
    SpatialAddressPairDataset,
    collate_spatial_pairs,
)


class DualEncoderAddressModel(nn.Module):
    """
    Dual-Encoder Embedding Model for mapping messy addresses and canonical geographic entities
    into a shared vector space with Mean-Pooling and L2-Normalization.
    """

    def __init__(self, backbone_name: str = config.EMBEDDING_BACKBONE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone_name)
        self.hidden_dim = self.encoder.config.hidden_size

    def mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(pooled, p=2, dim=1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encode(input_ids, attention_mask)

    @classmethod
    def from_pretrained(cls, model_dir, backbone_name: str = config.EMBEDDING_BACKBONE):
        model = cls(backbone_name=backbone_name)
        weights_path = Path(model_dir) / "pytorch_model.bin"
        if weights_path.exists():
            state_dict = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state_dict)
        return model


def multiple_negatives_ranking_loss(
    anchor_embs: torch.Tensor,
    pos_embs: torch.Tensor,
    scale: float = 20.0,
) -> torch.Tensor:
    """Multiple Negatives Ranking Loss using in-batch negatives."""
    scores = torch.matmul(anchor_embs, pos_embs.t()) * scale
    labels = torch.arange(anchor_embs.size(0), device=anchor_embs.device)
    return F.cross_entropy(scores, labels)


def evaluate_spatial_validation(
    model: DualEncoderAddressModel,
    tokenizer: AutoTokenizer,
    val_pairs: List[Dict[str, Any]],
    all_canonical_entities: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, float]:
    """Validation retrieval check against canonical index."""
    model.eval()
    cand_texts = [e["canonical_text"] for e in all_canonical_entities]

    cand_enc = tokenizer(
        cand_texts,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        cand_vectors = model.encode(cand_enc["input_ids"], cand_enc["attention_mask"])

    queries = [p["raw_address"] for p in val_pairs[:300]]
    targets = [p["canonical_target"] for p in val_pairs[:300]]

    q_enc = tokenizer(
        queries,
        padding=True,
        truncation=True,
        max_length=96,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        q_vectors = model.encode(q_enc["input_ids"], q_enc["attention_mask"])
        sims = torch.matmul(q_vectors, cand_vectors.t())

    top1_correct = 0
    mrr_sum = 0.0

    for i, target_text in enumerate(targets):
        row_sims = sims[i]
        sorted_indices = torch.argsort(row_sims, descending=True).cpu().tolist()
        ranked_texts = [cand_texts[idx] for idx in sorted_indices]

        if ranked_texts[0] == target_text:
            top1_correct += 1

        if target_text in ranked_texts:
            rank = ranked_texts.index(target_text) + 1
            mrr_sum += 1.0 / rank

    n = len(queries)
    return {
        "val_recall_at_1": round(top1_correct / n, 4) if n else 0.0,
        "val_mrr": round(mrr_sum / n, 4) if n else 0.0,
    }


def train_spatial_model():
    set_seed(config.RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Device name: {torch.cuda.get_device_name(0)}")

    # 1. Load splits & canonical database
    train_pairs, val_pairs, test_pairs = load_spatial_training_splits()
    _, _, all_canonical_entities = load_canonical_spatial_database()

    print(f"\n[+] Loaded Spatial Training Pairs (Zero Leakage):")
    print(f"    Train pairs: {len(train_pairs)} (Locality + Landmark POI targets)")
    print(f"    Val pairs:   {len(val_pairs)}")
    print(f"    Test pairs:  {len(test_pairs)} (strictly held-out)")
    print(f"    Canonical Index Size: {len(all_canonical_entities)} (36 Localities + 240 POIs)")

    # 2. Tokenizer & Model Backbone
    tokenizer = AutoTokenizer.from_pretrained(config.EMBEDDING_BACKBONE)
    model = DualEncoderAddressModel(backbone_name=config.EMBEDDING_BACKBONE).to(device)

    # 3. Augmenter
    augmenter = AddressAugmenter(
        augmentation_prob=config.AUGMENTATION_PROB,
        shuffle_prob=config.SHUFFLE_PROB,
        noise_prob=config.NOISE_PROB,
    )

    train_dataset = SpatialAddressPairDataset(
        train_pairs,
        tokenizer=tokenizer,
        augmenter=augmenter,
        is_training=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda b: collate_spatial_pairs(b, tokenizer),
    )

    # 4. Optimizer & Warmup Scheduler
    optimizer = AdamW(model.parameters(), lr=config.EMBEDDING_LEARNING_RATE, weight_decay=0.01)
    total_steps = len(train_loader) * config.EMBEDDING_NUM_EPOCHS
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    save_dir = config.SPATIAL_EMBEDDING_MODEL_V1_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    best_mrr = 0.0

    print(f"\n[+] Starting Dual-Encoder Spatial Metric Learning for {config.EMBEDDING_NUM_EPOCHS} epochs...")

    for epoch in range(1, config.EMBEDDING_NUM_EPOCHS + 1):
        model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.EMBEDDING_NUM_EPOCHS}")
        for batch in pbar:
            optimizer.zero_grad()

            anchor_ids = batch["anchor_input_ids"].to(device)
            anchor_mask = batch["anchor_attention_mask"].to(device)
            pos_ids = batch["pos_input_ids"].to(device)
            pos_mask = batch["pos_attention_mask"].to(device)

            anchor_embs = model(anchor_ids, anchor_mask)
            pos_embs = model(pos_ids, pos_mask)

            loss = multiple_negatives_ranking_loss(anchor_embs, pos_embs, scale=20.0)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / len(train_loader)

        val_metrics = evaluate_spatial_validation(
            model=model,
            tokenizer=tokenizer,
            val_pairs=val_pairs,
            all_canonical_entities=all_canonical_entities,
            device=device,
        )
        print(f"Epoch {epoch} | Loss: {avg_loss:.4f} | Val Recall@1: {val_metrics['val_recall_at_1']:.4f} | Val MRR: {val_metrics['val_mrr']:.4f}")

        if val_metrics["val_mrr"] > best_mrr:
            best_mrr = val_metrics["val_mrr"]
            print(f"  -> New best validation MRR ({val_metrics['val_mrr']:.4f})! Saving checkpoint to {save_dir}...")
            torch.save(model.state_dict(), save_dir / "pytorch_model.bin")
            tokenizer.save_pretrained(save_dir)

    print(f"\n[+] Training Complete! Final spatial model checkpoint saved to: {save_dir}")


if __name__ == "__main__":
    train_spatial_model()
