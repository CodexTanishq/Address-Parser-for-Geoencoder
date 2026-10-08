import json
import argparse
from pathlib import Path
from typing import Dict, Any
import torch
from transformers import AutoTokenizer

from src.embedding_model import config
from src.embedding_model.model import HierarchicalAddressEncoder
from src.embedding_model.hierarchy import GazetteerIndex
from src.embedding_model.retrieval import HierarchicalRetriever
from src.embedding_model.preprocessing import preprocess_address


class AddressEmbeddingInferenceEngine:
    """
    End-to-End Hierarchical Address Retrieval Inference Engine.
    Given raw address -> preprocessing -> address vector -> hierarchical candidate retrieval
    Returns structured JSON with top-k candidates, similarity scores, rank margins, and confidence flags.
    """
    def __init__(
        self,
        checkpoint_path: str = str(config.MODEL_SAVE_DIR / "best_model.pt"),
        device: torch.device = config.DEVICE
    ):
        self.device = device
        self.gazetteer = GazetteerIndex()

        # Load Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)

        # Build candidate dictionaries
        candidate_towns = list(self.gazetteer.town_names)
        candidate_localities = {}
        for tid, loc_list in self.gazetteer.town_to_localities.items():
            tname = self.gazetteer.towns.get(tid, "")
            candidate_localities[tname] = [(self.gazetteer.localities[lid][0], lid) for lid in loc_list]

        candidate_landmarks = {}
        for lname, lms in self.gazetteer.locality_to_landmarks.items():
            candidate_landmarks[lname] = list(lms)

        # Load Model
        self.model = HierarchicalAddressEncoder(
            model_name=config.PRETRAINED_MODEL_NAME,
            projection_dim=config.PROJECTION_DIM,
            pooling_mode=config.POOLING_MODE,
        ).to(self.device)

        cp = Path(checkpoint_path)
        if cp.exists():
            print(f"Loading checkpoint from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
        else:
            print(f"Warning: Checkpoint {checkpoint_path} not found. Running with un-finetuned weights.")

        self.model.eval()

        # Initialize Retriever
        self.retriever = HierarchicalRetriever(
            model=self.model,
            tokenizer=self.tokenizer,
            gazetteer=self.gazetteer,
            device=self.device,
            candidate_towns=candidate_towns,
            candidate_localities=candidate_localities,
            candidate_landmarks=candidate_landmarks
        )

    def predict(
        self,
        raw_address: str,
        top_k: int = 3,
        preprocessing_mode: str = config.PREPROCESSING_MODE,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
        margin_threshold: float = config.MARGIN_THRESHOLD,
        global_locality: bool = False
    ) -> Dict[str, Any]:
        """
        Executes hierarchical prediction pipeline on raw address string.
        """
        # Step 1: Preprocessing
        clean_address = preprocess_address(raw_address, mode=preprocessing_mode)

        # Step 2: Address Embedding
        encoded = self.tokenizer(
            [clean_address],
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            address_emb = self.model.encode_address(encoded["input_ids"], encoded["attention_mask"])

        # Step 3: Hierarchical Retrieval
        retrieval_res = self.retriever.retrieve(
            address_emb=address_emb,
            top_k=top_k,
            similarity_threshold=similarity_threshold,
            margin_threshold=margin_threshold,
            global_locality=global_locality
        )

        # Construct final output
        output = {
            "raw_address": raw_address,
            "preprocessed_address": clean_address,
            "town": {
                "prediction": retrieval_res["predicted_town"],
                "similarity": retrieval_res["town_similarity"],
                "margin": retrieval_res["town_margin"],
                "is_confident": retrieval_res["town_confident"]
            },
            "locality": {
                "prediction": retrieval_res["predicted_locality"],
                "similarity": retrieval_res["locality_similarity"],
                "margin": retrieval_res["locality_margin"],
                "is_confident": retrieval_res["locality_confident"]
            },
            "landmark": {
                "prediction": retrieval_res["predicted_landmark"],
                "similarity": retrieval_res["landmark_similarity"],
                "margin": retrieval_res["landmark_margin"],
                "is_confident": retrieval_res["landmark_confident"]
            },
            "candidates": {
                "towns": retrieval_res["top_towns"],
                "localities": retrieval_res["top_localities"],
                "landmarks": retrieval_res["top_landmarks"]
            }
        }
        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical Address Embedding Inference")
    parser.add_argument("--address", type=str, default="6th Cross, 5th Main, near church, Kuvempu Layt, Kaveripura - 960102")
    args = parser.parse_args()

    engine = AddressEmbeddingInferenceEngine()
    res = engine.predict(args.address)
    print(json.dumps(res, indent=2, ensure_ascii=False))
