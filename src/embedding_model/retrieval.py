import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
from src.embedding_model import config
from src.embedding_model.hierarchy import GazetteerIndex


class HierarchicalRetriever:
    """
    Performs multi-stage hierarchical candidate retrieval in the shared metric space:
    1. Town Retrieval: compare address embedding against all candidate town embeddings
    2. Locality Retrieval: restrict search to localities belonging to the predicted town
    3. Landmark Retrieval: restrict search to landmarks associated with predicted locality
    4. Confidence Margin & Calibration: compute rank margin between top-1 and top-2 candidates
    """
    def __init__(
        self,
        model,
        tokenizer,
        gazetteer: GazetteerIndex,
        device: torch.device,
        candidate_towns: List[str],
        candidate_localities: Dict[str, List[Tuple[str, str]]],  # town_name -> list of (loc_name, loc_id)
        candidate_landmarks: Dict[str, List[str]],  # loc_name -> list of landmarks
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.gazetteer = gazetteer
        self.device = device
        self.candidate_towns = candidate_towns
        self.candidate_localities = candidate_localities
        self.candidate_landmarks = candidate_landmarks

        # Precompute entity embeddings for fast vector retrieval
        self.precompute_entity_embeddings()

    def _encode_text(self, text_list: List[str]) -> torch.Tensor:
        encoded = self.tokenizer(
            text_list,
            padding=True,
            truncation=True,
            max_length=config.MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            embs = self.model.encode_entity(encoded["input_ids"], encoded["attention_mask"])
        return embs  # [N, D]

    def precompute_entity_embeddings(self):
        self.model.eval()
        # 1. Town embeddings
        self.town_embs = self._encode_text(self.candidate_towns)  # [T, D]

        # 2. Locality embeddings with town context
        self.loc_embs_by_town = {}
        for town_name, loc_tuples in self.candidate_localities.items():
            if not loc_tuples:
                continue
            # Contextual entity string: "Locality Name, Town Name"
            context_strings = [f"{lname}, {town_name}" for lname, lid in loc_tuples]
            self.loc_embs_by_town[town_name] = self._encode_text(context_strings)

        # 3. Landmark embeddings with locality context
        self.landmark_embs_by_loc = {}
        for loc_name, landmarks in self.candidate_landmarks.items():
            if not landmarks:
                continue
            context_strings = [f"{lm}, {loc_name}" for lm in landmarks]
            self.landmark_embs_by_loc[loc_name] = self._encode_text(context_strings)

    def retrieve(
        self,
        address_emb: torch.Tensor,
        top_k: int = 3,
        similarity_threshold: float = config.SIMILARITY_THRESHOLD,
        margin_threshold: float = config.MARGIN_THRESHOLD,
        global_locality: bool = False,
    ) -> Dict[str, Any]:
        """
        address_emb: [1, D] L2-normalized vector
        global_locality: if True, performs flat unconditioned locality search (ablation M vs N)
        """
        # --- Stage 1: Town Retrieval ---
        # Cosine similarity: [1, D] @ [T, D]^T -> [T]
        town_sims = torch.matmul(address_emb, self.town_embs.t()).squeeze(0).cpu().tolist()
        sorted_town_indices = sorted(range(len(town_sims)), key=lambda i: town_sims[i], reverse=True)

        top_towns = [
            {"name": self.candidate_towns[i], "similarity": round(float(town_sims[i]), 4), "rank": r + 1}
            for r, i in enumerate(sorted_town_indices[:top_k])
        ]
        best_town = top_towns[0]
        town_margin = round(top_towns[0]["similarity"] - top_towns[1]["similarity"], 4) if len(top_towns) > 1 else 1.0
        best_town["margin"] = town_margin
        best_town["is_confident"] = (best_town["similarity"] >= similarity_threshold and town_margin >= margin_threshold)

        # --- Stage 2: Locality Retrieval ---
        if global_locality:
            # Flatten all localities across all towns
            all_loc_tuples = []
            all_loc_embs_list = []
            for tname, l_tuples in self.candidate_localities.items():
                if tname in self.loc_embs_by_town:
                    all_loc_tuples.extend([(tname, lt[0], lt[1]) for lt in l_tuples])
                    all_loc_embs_list.append(self.loc_embs_by_town[tname])
            all_loc_embs = torch.cat(all_loc_embs_list, dim=0)
            loc_sims = torch.matmul(address_emb, all_loc_embs.t()).squeeze(0).cpu().tolist()
            sorted_loc_indices = sorted(range(len(loc_sims)), key=lambda i: loc_sims[i], reverse=True)
            top_localities = [
                {"name": all_loc_tuples[i][1], "locality_id": all_loc_tuples[i][2], "town": all_loc_tuples[i][0], "similarity": round(float(loc_sims[i]), 4), "rank": r + 1}
                for r, i in enumerate(sorted_loc_indices[:top_k])
            ]
        else:
            # Hierarchical: Conditioned strictly on the predicted top-1 town
            predicted_town_name = best_town["name"]
            loc_tuples = self.candidate_localities.get(predicted_town_name, [])
            if loc_tuples and predicted_town_name in self.loc_embs_by_town:
                target_loc_embs = self.loc_embs_by_town[predicted_town_name]
                loc_sims = torch.matmul(address_emb, target_loc_embs.t()).squeeze(0).cpu().tolist()
                sorted_loc_indices = sorted(range(len(loc_sims)), key=lambda i: loc_sims[i], reverse=True)
                top_localities = [
                    {"name": loc_tuples[i][0], "locality_id": loc_tuples[i][1], "similarity": round(float(loc_sims[i]), 4), "rank": r + 1}
                    for r, i in enumerate(sorted_loc_indices[:top_k])
                ]
            else:
                top_localities = [{"name": "UNKNOWN", "locality_id": "UNKNOWN", "similarity": 0.0, "rank": 1}]

        best_loc = top_localities[0]
        loc_margin = round(top_localities[0]["similarity"] - top_localities[1]["similarity"], 4) if len(top_localities) > 1 else 1.0
        best_loc["margin"] = loc_margin
        best_loc["is_confident"] = (best_loc["similarity"] >= similarity_threshold and loc_margin >= margin_threshold)

        # --- Stage 3: Landmark Retrieval ---
        # Conditioned on predicted locality
        predicted_loc_name = best_loc["name"]
        landmark_candidates = self.candidate_landmarks.get(predicted_loc_name, [])
        if landmark_candidates and predicted_loc_name in self.landmark_embs_by_loc:
            target_lm_embs = self.landmark_embs_by_loc[predicted_loc_name]
            lm_sims = torch.matmul(address_emb, target_lm_embs.t()).squeeze(0).cpu().tolist()
            sorted_lm_indices = sorted(range(len(lm_sims)), key=lambda i: lm_sims[i], reverse=True)
            top_landmarks = [
                {"name": landmark_candidates[i], "similarity": round(float(lm_sims[i]), 4), "rank": r + 1}
                for r, i in enumerate(sorted_lm_indices[:top_k])
            ]
        else:
            top_landmarks = [{"name": "UNKNOWN", "similarity": 0.0, "rank": 1}]

        best_lm = top_landmarks[0]
        lm_margin = round(top_landmarks[0]["similarity"] - top_landmarks[1]["similarity"], 4) if len(top_landmarks) > 1 else 1.0
        best_lm["margin"] = lm_margin
        best_lm["is_confident"] = (best_lm["similarity"] >= similarity_threshold and lm_margin >= margin_threshold)

        return {
            "town": {
                "prediction": best_town["name"] if best_town["is_confident"] else "LOW_CONFIDENCE",
                "similarity": best_town["similarity"],
                "margin": best_town["margin"],
                "top_k": top_towns
            },
            "locality": {
                "prediction": best_loc["name"] if best_loc["is_confident"] else "LOW_CONFIDENCE",
                "locality_id": best_loc.get("locality_id"),
                "similarity": best_loc["similarity"],
                "margin": best_loc["margin"],
                "top_k": top_localities
            },
            "landmark": {
                "prediction": best_lm["name"] if best_lm["is_confident"] else "LOW_CONFIDENCE",
                "similarity": best_lm["similarity"],
                "margin": best_lm["margin"],
                "top_k": top_landmarks
            }
        }
