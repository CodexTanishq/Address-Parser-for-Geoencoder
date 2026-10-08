"""
Script to evaluate the Spatial Dual-Encoder Embedding Model on the held-out test split.
Computes Recall@1, Recall@5, and MRR for both Locality and Landmark retrieval.
Saves metrics to outputs/embedding_retrieval_metrics.json and prints sample predictions.
"""
import json
import os
import torch
from transformers import AutoTokenizer
from tqdm import tqdm

from src.config import (
    SPATIAL_EMBEDDING_MODEL_V1_DIR,
    COORDINATE_INDEX_PATH,
    COORDINATE_METADATA_PKL,
    EMBEDDING_RETRIEVAL_METRICS_JSON
)
from src.embedding_dataset import load_spatial_training_splits
from src.train_embedding import DualEncoderAddressModel
from src.coordinate_retriever import SpatialCoordinateRetriever

def evaluate_retrieval():
    print("=" * 60)
    print("EVALUATING DUAL-ENCODER SPATIAL RETRIEVAL ON HELD-OUT TEST SPLIT")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load test split
    _, _, test_pairs = load_spatial_training_splits()
    print(f"Test Split Size: {len(test_pairs)} address-target pairs")
    
    # 2. Load trained model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(SPATIAL_EMBEDDING_MODEL_V1_DIR)
    model = DualEncoderAddressModel.from_pretrained(SPATIAL_EMBEDDING_MODEL_V1_DIR)
    model.to(device)
    model.eval()
    
    # 3. Load Coordinate Index and Metadata
    coordinate_index = torch.load(COORDINATE_INDEX_PATH, map_location=device) # [N, 768]
    import pickle
    with open(COORDINATE_METADATA_PKL, "rb") as f:
        metadata = pickle.load(f)
        
    # Map target string/name to canonical indices
    # We can match either by canonical_text directly or by entity name
    text_to_idx = {meta["canonical_text"]: idx for idx, meta in enumerate(metadata)}
    name_to_idx = {meta["name"].lower(): idx for idx, meta in enumerate(metadata)}
    
    # 4. Separate metrics by target_type (LOCALITY vs LANDMARK)
    metrics_by_type = {
        "LOCALITY": {"hits_at_1": 0, "hits_at_5": 0, "mrr_sum": 0.0, "total": 0},
        "LANDMARK": {"hits_at_1": 0, "hits_at_5": 0, "mrr_sum": 0.0, "total": 0},
        "OVERALL": {"hits_at_1": 0, "hits_at_5": 0, "mrr_sum": 0.0, "total": 0}
    }
    
    # Batch encode queries
    batch_size = 64
    all_query_embs = []
    
    print("Encoding test query addresses...")
    with torch.no_grad():
        for i in range(0, len(test_pairs), batch_size):
            batch = test_pairs[i : i + batch_size]
            queries = [p["raw_address"] for p in batch]
            encoded = tokenizer(
                queries,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)
            emb = model(encoded["input_ids"], encoded["attention_mask"])
            all_query_embs.append(emb)
            
    query_tensor = torch.cat(all_query_embs, dim=0) # [num_test, 768]
    
    # Compute similarity matrix: [num_test, num_canonical]
    print("Computing cosine similarity against all canonical spatial entities...")
    with torch.no_grad():
        sim_matrix = torch.matmul(query_tensor, coordinate_index.t()) # [num_test, N]
        sorted_indices = torch.argsort(sim_matrix, dim=1, descending=True).cpu()
        
    for i, pair in enumerate(test_pairs):
        target_text = pair.get("canonical_target", "")
        target_name = pair.get("target_name", "").lower()
        target_type = pair.get("target_type", "OVERALL")
        
        target_pos = text_to_idx.get(target_text)
        if target_pos is None:
            target_pos = name_to_idx.get(target_name)
        if target_pos is None:
            continue
            
        ranking = (sorted_indices[i] == target_pos).nonzero(as_tuple=True)[0]
        rank = int(ranking[0].item()) + 1 if len(ranking) > 0 else 9999
        
        # Update metrics
        for group_key in [target_type, "OVERALL"]:
            metrics_by_type[group_key]["total"] += 1
            if rank == 1:
                metrics_by_type[group_key]["hits_at_1"] += 1
            if rank <= 5:
                metrics_by_type[group_key]["hits_at_5"] += 1
            metrics_by_type[group_key]["mrr_sum"] += 1.0 / rank


    # Format summary
    summary = {}
    for k, v in metrics_by_type.items():
        total = v["total"]
        if total > 0:
            summary[k] = {
                "total_queries": total,
                "recall_at_1": round(v["hits_at_1"] / total, 4),
                "recall_at_5": round(v["hits_at_5"] / total, 4),
                "mrr": round(v["mrr_sum"] / total, 4)
            }
            
    print("\n" + "=" * 60)
    print("TEST EVALUATION RESULTS:")
    print("=" * 60)
    print(json.dumps(summary, indent=2))
    
    # Save metrics JSON
    os.makedirs(os.path.dirname(EMBEDDING_RETRIEVAL_METRICS_JSON), exist_ok=True)
    with open(EMBEDDING_RETRIEVAL_METRICS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[OK] Metrics saved to {EMBEDDING_RETRIEVAL_METRICS_JSON}")
    
    # 5. Test Inference Sample with Retriever
    print("\n" + "=" * 60)
    print("TEST SAMPLE RETRIEVAL WITH COORDINATES:")
    print("=" * 60)
    retriever = SpatialCoordinateRetriever()
    test_queries = [
        "Opp. Post Office, Kuvempu Layt, 560001",
        "Beside Bus Stand, Gandhi Bazar Main Rd, Shimoga",
        "Near Government School, Ward 12, Indiranagar, 560038"
    ]
    for q in test_queries:
        res = retriever.retrieve(q, top_k_landmarks=2)
        print(f"\nQuery: {res['query_address']}")
        print(f"Top Locality: {json.dumps(res['top_locality'], indent=2)}")
        print(f"Top Landmarks: {json.dumps(res['top_landmarks'], indent=2)}")

if __name__ == "__main__":
    evaluate_retrieval()
