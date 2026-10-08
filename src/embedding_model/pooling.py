import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    Learned attention-based pooling over contextual token representations:
    alpha_i = softmax(w^T tanh(W * h_i))
    address_rep = sum(alpha_i * h_i)
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1, bias=False)
        )

    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # token_embeddings: [batch_size, seq_len, hidden_dim]
        # attention_mask: [batch_size, seq_len]
        scores = self.attn_net(token_embeddings).squeeze(-1)  # [batch_size, seq_len]
        # Mask out padding tokens with large negative value
        scores = scores.masked_fill(attention_mask == 0, -1e9)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)  # [batch_size, seq_len, 1]
        pooled = torch.sum(token_embeddings * weights, dim=1)  # [batch_size, hidden_dim]
        return pooled


class MeanPooling(nn.Module):
    """Unweighted average of non-padding contextual token representations."""
    def forward(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask


class ProjectionHead(nn.Module):
    """Projects contextual representations into common metric space with L2 normalization."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.proj(x)
        return F.normalize(projected, p=2, dim=-1)
