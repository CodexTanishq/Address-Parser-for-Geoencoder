from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from src import config
from src.embedding_dataset import load_canonical_database
from src.train_embedding import DualEncoderAddressModel


class AddressEmbeddingPredictor:
    """
    Inference Engine for encoding noisy/permuted raw addresses into normalized vectors
    and retrieving the top-k most similar canonical localities and towns via Cosine Similarity.
    """

    def __init__(
        self,
        model_dir: Path = config.ADDRESS_EMBEDDING_MODEL_V1_DIR,
        device: Optional[torch.device] = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_dir = Path(model_dir)

        # 1. Load Tokenizer & Model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = DualEncoderAddressModel(backbone_name=config.EMBEDDING_BACKBONE)

        weights_path = self.model_dir / "pytorch_model.bin"
        if not weights_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found at {weights_path}!")

        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 2. Precompute Canonical Gazetteer Vectors
        self.canonical_df, self.canonical_lookup, _ = load_canonical_database()
        self._precompute_canonical_index()

    def _precompute_canonical_index(self):
        canonical_texts = self.canonical_df["canonical_text"].tolist()
        self.canonical_lids = self.canonical_df["locality_id"].tolist()

        enc = self.tokenizer(
            canonical_texts,
            padding=True,
            truncation=True,
            max_length=config.EMBEDDING_MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            self.canonical_vectors = self.model.encode(enc["input_ids"], enc["attention_mask"])  # [C, D]

    def encode(self, text: str) -> torch.Tensor:
        """Encodes an input address string into a 768-d unit-normalized vector."""
        enc = self.tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=config.EMBEDDING_MAX_SEQ_LENGTH,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            emb = self.model.encode(enc["input_ids"], enc["attention_mask"])
        return emb.squeeze(0)  # [D]

    def retrieve(self, address_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Given raw messy address text, returns:
        - top_1: Best predicted locality + town
        - candidates: List of top-k candidate entities with cosine similarity scores and confidence margin.
        """
        query_vec = self.encode(address_text).unsqueeze(0)  # [1, D]

        # Compute cosine similarity
        with torch.no_grad():
            cos_sims = torch.matmul(query_vec, self.canonical_vectors.t()).squeeze(0)  # [C]

        sorted_indices = torch.argsort(cos_sims, descending=True).cpu().tolist()[:top_k]

        candidates = []
        for rank, idx in enumerate(sorted_indices, 1):
            lid = self.canonical_lids[idx]
            meta = self.canonical_lookup[lid]
            sim = float(cos_sims[idx].item())
            candidates.append({
                "rank": rank,
                "locality_id": lid,
                "locality_name": meta["locality_name"],
                "town_name": meta["town_name"],
                "pincode": meta["pincode"],
                "canonical_name": meta["canonical_text"],
                "cosine_similarity": round(sim, 4),
            })

        top_1 = candidates[0]
        margin = round(candidates[0]["cosine_similarity"] - candidates[1]["cosine_similarity"], 4) if len(candidates) > 1 else 1.0

        return {
            "query_address": address_text,
            "top_1_prediction": {
                "locality_id": top_1["locality_id"],
                "locality_name": top_1["locality_name"],
                "town_name": top_1["town_name"],
                "canonical_name": top_1["canonical_name"],
                "confidence_score": top_1["cosine_similarity"],
                "margin_to_next": margin,
            },
            "top_k_candidates": candidates,
        }
