"""
Script to build the canonical spatial coordinate index using the trained Dual-Encoder model.
Encodes 36 canonical localities and 240 canonical landmarks/POIs into dense embeddings,
and stores the index and metadata for ultra-fast vector retrieval.
"""
import os
import pickle
import torch
from transformers import AutoTokenizer

from src.config import (
    SPATIAL_EMBEDDING_MODEL_V1_DIR,
    COORDINATE_INDEX_PATH,
    COORDINATE_METADATA_PKL
)
from src.embedding_dataset import load_canonical_spatial_database
from src.train_embedding import DualEncoderAddressModel

def build_coordinate_index(
    model_dir: str = SPATIAL_EMBEDDING_MODEL_V1_DIR,
    index_path: str = COORDINATE_INDEX_PATH,
    metadata_path: str = COORDINATE_METADATA_PKL
):
    print("=" * 60)
    print("BUILDING CANONICAL SPATIAL COORDINATE INDEX")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Load trained model & tokenizer
    print(f"Loading dual-encoder model from: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = DualEncoderAddressModel.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # 2. Load canonical database (Localities + Landmarks)
    _, _, all_entities = load_canonical_spatial_database()
    print(f"Loaded {len(all_entities)} canonical spatial entities.")
    
    canonical_texts = [e["canonical_text"] for e in all_entities]
    metadata = all_entities
    
    # 3. Batch encode entities
    batch_size = 64
    all_embeddings = []
    
    print(f"Encoding {len(canonical_texts)} canonical entities...")
    with torch.no_grad():
        for i in range(0, len(canonical_texts), batch_size):
            batch_texts = canonical_texts[i : i + batch_size]
            encoded = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt"
            ).to(device)
            
            embeddings = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            )
            all_embeddings.append(embeddings.cpu())
            
    index_tensor = torch.cat(all_embeddings, dim=0) # [N, 768] (L2 normalized)
    print(f"Index tensor shape: {index_tensor.shape}")
    
    # 4. Save index and metadata
    os.makedirs(os.path.dirname(index_path), exist_ok=True)
    
    torch.save(index_tensor, index_path)
    print(f"[OK] Saved spatial coordinate index tensor to: {index_path}")
    
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"[OK] Saved spatial metadata to: {metadata_path}")
    
    print("Coordinate index build completed successfully!")
    return index_tensor, metadata

if __name__ == "__main__":
    build_coordinate_index()
