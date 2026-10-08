"""
Module for Spatial Coordinate Retrieval using the trained Dual-Encoder model.
Given a raw or noisy address string, encodes it into dense embedding space,
retrieves the top-K closest matching canonical localities and landmarks,
and returns their exact (x, y) coordinates for Bayesian Fusion.
"""
from typing import Dict, Any, List, Optional
import os
import pickle
import torch
from transformers import AutoTokenizer

from src.config import (
    SPATIAL_EMBEDDING_MODEL_V1_DIR,
    COORDINATE_INDEX_PATH,
    COORDINATE_METADATA_PKL
)
from src.train_embedding import DualEncoderAddressModel

class SpatialCoordinateRetriever:
    """
    Dual-Encoder Spatial Coordinate Retriever for Locality and Landmark resolution.
    Retrieves canonical geographic entities and outputs exact coordinates.
    """
    def __init__(
        self,
        model_dir: str = str(SPATIAL_EMBEDDING_MODEL_V1_DIR),
        index_path: str = str(COORDINATE_INDEX_PATH),
        metadata_path: str = str(COORDINATE_METADATA_PKL),
        device: Optional[str] = None
    ):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        # 1. Load Tokenizer & Dual-Encoder Model
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = DualEncoderAddressModel.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        
        # 2. Load Coordinate Index Tensor & Metadata
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Coordinate index or metadata not found. Expected {index_path} and {metadata_path}. "
                "Please run `src/build_coordinate_index.py` first."
            )
            
        self.index_tensor = torch.load(index_path, map_location=self.device) # [N, 768] (normalized)
        with open(metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
            
        assert len(self.index_tensor) == len(self.metadata), (
            f"Index dimension mismatch: {len(self.index_tensor)} vectors vs {len(self.metadata)} records"
        )

    def encode_query(self, address_text: str) -> torch.Tensor:
        """Encodes an address string to normalized embedding vector."""
        encoded = self.tokenizer(
            [address_text],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            emb = self.model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"]
            ) # [1, 768]
        return emb

    def retrieve(
        self,
        address_text: str,
        top_k_landmarks: int = 3
    ) -> Dict[str, Any]:
        """
        Retrieves top locality and top landmarks with coordinates for an address query.
        
        Returns:
            dict containing:
                - query_address
                - top_locality: dict(locality_name, town_name, centroid_x, centroid_y, similarity)
                - top_landmarks: list of dicts(landmark_name, landmark_type, x, y, similarity)
        """
        query_emb = self.encode_query(address_text) # [1, 768]
        
        # Cosine similarity across all canonical entities
        similarities = torch.matmul(query_emb, self.index_tensor.t()).squeeze(0) # [N]
        scores, indices = torch.sort(similarities, descending=True)
        
        scores = scores.cpu().tolist()
        indices = indices.cpu().tolist()
        
        top_locality = None
        top_landmarks = []
        
        for idx, score in zip(indices, scores):
            meta = self.metadata[idx]
            entity_type = meta["entity_type"]
            
            if entity_type == "LOCALITY" and top_locality is None:
                top_locality = {
                    "locality_id": meta["entity_id"],
                    "locality_name": meta["name"],
                    "town_name": meta.get("town_name", ""),
                    "centroid_x": float(meta["x"]),
                    "centroid_y": float(meta["y"]),
                    "similarity": round(float(score), 4)
                }
            elif entity_type == "LANDMARK" and len(top_landmarks) < top_k_landmarks:
                top_landmarks.append({
                    "poi_id": meta["entity_id"],
                    "landmark_name": meta["name"],
                    "landmark_type": meta.get("landmark_type", ""),
                    "town_name": meta.get("town_name", ""),
                    "x": float(meta["x"]),
                    "y": float(meta["y"]),
                    "similarity": round(float(score), 4)
                })
                
            # If we collected top locality and required landmarks, stop early
            if top_locality is not None and len(top_landmarks) >= top_k_landmarks:
                break
                
        return {
            "query_address": address_text,
            "top_locality": top_locality,
            "top_landmarks": top_landmarks
        }
