import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Optional, Tuple, List

from src.embedding_model import config
from src.embedding_model.pooling import AttentionPooling, MeanPooling, ProjectionHead


class HierarchicalAddressEncoder(nn.Module):
    """
    Hierarchical Address & Entity Representation Model.
    Directly maps raw addresses, town context, locality context, and landmark context
    into a unified metric space for hierarchical candidate retrieval.
    """
    def __init__(
        self,
        model_name: str = config.PRETRAINED_MODEL_NAME,
        projection_dim: int = config.PROJECTION_DIM,
        pooling_mode: str = config.POOLING_MODE,
    ):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_dim = self.encoder.config.hidden_size
        self.pooling_mode = pooling_mode

        # Address Pooling Mechanism
        if pooling_mode == "attention":
            self.address_pooler = AttentionPooling(hidden_dim)
        elif pooling_mode == "mean":
            self.address_pooler = MeanPooling()
        elif pooling_mode == "cls":
            self.address_pooler = None
        else:
            raise ValueError(f"Unknown pooling mode: {pooling_mode}")

        # Entity Pooling (Mean pooling over short entity strings)
        self.entity_pooler = MeanPooling()

        # Shared/Symmetric Projection Heads into the common space
        self.address_projection = ProjectionHead(hidden_dim, projection_dim)
        self.entity_projection = ProjectionHead(hidden_dim, projection_dim)

        # Hierarchy Compatibility Layer (Parent-Child composition)
        self.locality_composition = nn.Linear(projection_dim * 2, projection_dim)
        self.landmark_composition = nn.Linear(projection_dim * 2, projection_dim)

    def encode_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state, outputs.last_hidden_state[:, 0, :]  # [B, L, D], [B, D] CLS

    def encode_address(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode raw address text into L2-normalized embedding vector."""
        token_embs, cls_emb = self.encode_tokens(input_ids, attention_mask)
        if self.pooling_mode == "attention":
            pooled = self.address_pooler(token_embs, attention_mask)
        elif self.pooling_mode == "mean":
            pooled = self.address_pooler(token_embs, attention_mask)
        elif self.pooling_mode == "cls":
            pooled = cls_emb
        return self.address_projection(pooled)

    def encode_entity(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode entity text (town, locality, or landmark) into common metric space."""
        token_embs, _ = self.encode_tokens(input_ids, attention_mask)
        pooled = self.entity_pooler(token_embs, attention_mask)
        return self.entity_projection(pooled)

    def compose_locality(self, locality_emb: torch.Tensor, town_emb: torch.Tensor) -> torch.Tensor:
        """f_loc(L, T) = L2_Norm(W * [L || T])"""
        combined = torch.cat([locality_emb, town_emb], dim=-1)
        composed = self.locality_composition(combined)
        return F.normalize(composed, p=2, dim=-1)

    def compose_landmark(self, landmark_emb: torch.Tensor, locality_emb: torch.Tensor) -> torch.Tensor:
        """f_land(M, L) = L2_Norm(W * [M || L])"""
        combined = torch.cat([landmark_emb, locality_emb], dim=-1)
        composed = self.landmark_composition(combined)
        return F.normalize(composed, p=2, dim=-1)
