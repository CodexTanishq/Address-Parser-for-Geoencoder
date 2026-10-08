import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import torch
from transformers import AutoTokenizer

from src import config
from src.embedding_dataset import load_embedding_splits, load_canonical_database
from src.train_embedding import DualEncoderAddressModel, evaluate_retrieval


def run_test_evaluation(
    model_dir: Path = config.ADDRESS_EMBEDDING_MODEL_V1_DIR,
    metrics_output_path: Path = config.EMBEDDING_METRICS_JSON,
) -> Dict[str, Any]:
    """
    Evaluates the trained dual-encoder embedding model strictly on the held-out TEST split.
    Calculates:
    - Top-1 Accuracy (Recall@1)
    - Top-5 Accuracy (Recall@5)
    - Mean Reciprocal Rank (MRR)
    - Average Cosine Similarity between noisy address vectors and true entity vectors.
    """
    print("=" * 75)
    print("  EVALUATING DUAL-ENCODER EMBEDDING MODEL ON HELD-OUT TEST SPLIT")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load test split
    _, _, test_records, canonical_df = load_embedding_splits()
    print(f"Held-out Test addresses: {len(test_records)}")
    print(f"Canonical entity targets: {len(canonical_df)}")

    # 2. Load model checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = DualEncoderAddressModel(backbone_name=config.EMBEDDING_BACKBONE)

    weights_path = model_dir / "pytorch_model.bin"
    if not weights_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found at {weights_path}!")

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Compute metrics
    metrics = evaluate_retrieval(
        model=model,
        tokenizer=tokenizer,
        records=test_records,
        canonical_df=canonical_df,
        device=device,
        top_k=5,
    )

    metrics_formatted = {
        "model_architecture": "DualEncoderAddressModel (google/muril-base-cased)",
        "embedding_dimension": model.hidden_dim,
        "test_sample_count": len(test_records),
        "canonical_entities_count": len(canonical_df),
        "recall_at_1": metrics["recall_at_1"],
        "recall_at_5": metrics["recall_at_5"],
        "mean_reciprocal_rank_mrr": metrics["mrr"],
        "avg_target_cosine_similarity": metrics["avg_target_cos_sim"],
    }

    print("\n" + "-" * 75)
    print("  HELD-OUT TEST SET EVALUATION METRICS")
    print("-" * 75)
    print(f"Top-1 Accuracy (Recall@1)            : {metrics_formatted['recall_at_1'] * 100:.2f}%")
    print(f"Top-5 Accuracy (Recall@5)            : {metrics_formatted['recall_at_5'] * 100:.2f}%")
    print(f"Mean Reciprocal Rank (MRR)           : {metrics_formatted['mean_reciprocal_rank_mrr']:.4f}")
    print(f"Avg Cosine Sim with Target Entity    : {metrics_formatted['avg_target_cosine_similarity']:.4f}")
    print("-" * 75)

    # 4. Save metrics to JSON
    metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output_path, "w", encoding="utf-8") as f:
        json.dump(metrics_formatted, f, indent=2)
    print(f"Metrics saved to {metrics_output_path}\n")

    return metrics_formatted


if __name__ == "__main__":
    run_test_evaluation()
