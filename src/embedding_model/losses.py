import torch
import torch.nn as nn
import torch.nn.functional as F
from src.embedding_model import config


def info_nce_loss(
    query_embs: torch.Tensor,
    positive_embs: torch.Tensor,
    negative_embs: torch.Tensor,
    temperature: float = config.TEMPERATURE,
) -> torch.Tensor:
    """
    InfoNCE Contrastive Loss with In-Batch and Explicit Hard Negatives:
    L = -log( exp(sim(q, p+) / T) / (exp(sim(q, p+) / T) + sum(exp(sim(q, n-) / T))) )
    
    query_embs: [B, D] (Normalized)
    positive_embs: [B, D] (Normalized)
    negative_embs: [B, K, D] or [B, D] (Normalized hard negatives)
    """
    # Positive logits: [B, 1]
    pos_sim = torch.sum(query_embs * positive_embs, dim=-1, keepdim=True) / temperature

    if negative_embs.dim() == 2:
        negative_embs = negative_embs.unsqueeze(1)  # [B, 1, D]

    # Hard negative logits: [B, K]
    # query_embs: [B, 1, D] * negative_embs: [B, K, D] -> sum over D
    hard_neg_sim = torch.sum(query_embs.unsqueeze(1) * negative_embs, dim=-1) / temperature

    # In-batch cross negatives: query against all other positives in batch
    in_batch_sim = torch.matmul(query_embs, positive_embs.t()) / temperature
    # Mask out the diagonal (positive pairs)
    diag_mask = torch.eye(query_embs.size(0), dtype=torch.bool, device=query_embs.device)
    in_batch_neg = in_batch_sim.masked_fill(diag_mask, -1e9)

    # Concatenate all negative logits
    all_neg_sim = torch.cat([hard_neg_sim, in_batch_neg], dim=-1)

    # Denominator: log(exp(pos) + sum(exp(neg)))
    logits = torch.cat([pos_sim, all_neg_sim], dim=-1)
    labels = torch.zeros(query_embs.size(0), dtype=torch.long, device=query_embs.device)
    return F.cross_entropy(logits, labels)


def triplet_margin_loss(
    query_embs: torch.Tensor,
    positive_embs: torch.Tensor,
    negative_embs: torch.Tensor,
    margin: float = config.TRIPLET_MARGIN,
) -> torch.Tensor:
    """Standard Triplet Loss: max(0, d(a, p) - d(a, n) + margin) using cosine distance."""
    pos_dist = 1.0 - torch.sum(query_embs * positive_embs, dim=-1)
    if negative_embs.dim() == 3:
        # Take hardest negative (minimum distance)
        neg_dist = 1.0 - torch.max(torch.sum(query_embs.unsqueeze(1) * negative_embs, dim=-1), dim=-1)[0]
    else:
        neg_dist = 1.0 - torch.sum(query_embs * negative_embs, dim=-1)
    return F.relu(pos_dist - neg_dist + margin).mean()


def hierarchy_compatibility_loss(
    locality_composed_embs: torch.Tensor,
    town_embs: torch.Tensor,
    margin: float = 0.2,
) -> torch.Tensor:
    """
    Penalizes parent-child vector divergence:
    Children should reside in a compatible geographic cone of their parent.
    """
    sim = torch.sum(locality_composed_embs * town_embs, dim=-1)
    # Loss incurred if child diverges below margin from parent
    return F.relu(1.0 - sim - margin).mean()
