import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import AutoTokenizer, AutoModel

from src.embedding_model import config
from src.embedding_model.hierarchy import GazetteerIndex
from src.embedding_model.preprocessing import preprocess_address


class TFIDFBaselineRetriever:
    """
    Non-neural baseline using TF-IDF and cosine similarity.
    Evaluates:
    1. Word-level TF-IDF
    2. Character n-gram TF-IDF (char n-grams 2-5)
    """
    def __init__(
        self,
        gazetteer: GazetteerIndex,
        ngram_type: str = "word",  # "word" or "char"
        preprocessing_mode: str = "normalized_transliterated"
    ):
        self.gazetteer = gazetteer
        self.ngram_type = ngram_type
        self.preprocessing_mode = preprocessing_mode

        if ngram_type == "char":
            self.vectorizer = TfidfVectorizer(
                analyzer="char_wb",
                ngram_range=(2, 5),
                lowercase=True,
                max_features=50000
            )
        else:
            self.vectorizer = TfidfVectorizer(
                analyzer="word",
                ngram_range=(1, 2),
                lowercase=True,
                max_features=50000
            )

        self._fit_and_index()

    def _fit_and_index(self):
        # Collect all entity strings to fit vectorizer vocabulary
        all_texts = []
        self.town_names = list(self.gazetteer.town_names)
        all_texts.extend([preprocess_address(t, self.preprocessing_mode) for t in self.town_names])

        self.locality_contexts = []
        self.locality_town_map = []
        for tid, loc_list in self.gazetteer.town_to_localities.items():
            tname = self.gazetteer.towns.get(tid, "")
            for lid in loc_list:
                lname = self.gazetteer.localities[lid][0]
                ctx = f"{lname}, {tname}"
                self.locality_contexts.append(preprocess_address(ctx, self.preprocessing_mode))
                self.locality_town_map.append((lname, tname))

        all_texts.extend(self.locality_contexts)
        self.vectorizer.fit(all_texts)

        # Precompute candidate matrices
        town_preprocessed = [preprocess_address(t, self.preprocessing_mode) for t in self.town_names]
        self.town_matrix = self.vectorizer.transform(town_preprocessed)
        self.locality_matrix = self.vectorizer.transform(self.locality_contexts)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        town_top1 = 0
        town_top3 = 0
        town_mrr = 0.0

        loc_top1 = 0
        loc_top3 = 0
        loc_mrr = 0.0
        loc_eval_count = 0

        hier_exact = 0

        for _, row in test_df.iterrows():
            addr_text = preprocess_address(str(row["address_text"]), self.preprocessing_mode)
            q_vec = self.vectorizer.transform([addr_text])

            true_town = str(row.get("town_name", ""))
            true_loc = str(row.get("locality_name", ""))

            # 1. Town Retrieval
            sims = cosine_similarity(q_vec, self.town_matrix)[0]
            ranked_indices = np.argsort(-sims)
            ranked_towns = [self.town_names[i] for i in ranked_indices]

            if true_town in ranked_towns:
                rank = ranked_towns.index(true_town) + 1
                if rank == 1:
                    town_top1 += 1
                if rank <= 3:
                    town_top3 += 1
                town_mrr += 1.0 / rank

            pred_town = ranked_towns[0]

            # 2. Locality Retrieval (Hierarchical: restricted to candidate town)
            if true_loc and true_loc != "UNKNOWN" and true_loc != "nan":
                loc_eval_count += 1
                # Filter localities by predicted town
                cand_indices = [
                    i for i, (lname, tname) in enumerate(self.locality_town_map)
                    if tname == pred_town
                ]
                if not cand_indices:
                    cand_indices = list(range(len(self.locality_town_map)))

                sub_matrix = self.locality_matrix[cand_indices]
                loc_sims = cosine_similarity(q_vec, sub_matrix)[0]
                sub_ranked = np.argsort(-loc_sims)
                ranked_locs = [self.locality_town_map[cand_indices[i]][0] for i in sub_ranked]

                if true_loc in ranked_locs:
                    rank = ranked_locs.index(true_loc) + 1
                    if rank == 1:
                        loc_top1 += 1
                        if pred_town == true_town:
                            hier_exact += 1
                    if rank <= 3:
                        loc_top3 += 1
                    loc_mrr += 1.0 / rank

        n_total = len(test_df)
        return {
            "town_top1": round(town_top1 / max(1, n_total), 4),
            "town_top3": round(town_top3 / max(1, n_total), 4),
            "town_mrr": round(town_mrr / max(1, n_total), 4),
            "locality_top1": round(loc_top1 / max(1, loc_eval_count), 4),
            "locality_top3": round(loc_top3 / max(1, loc_eval_count), 4),
            "locality_mrr": round(loc_mrr / max(1, loc_eval_count), 4),
            "hierarchical_exact_match": round(hier_exact / max(1, loc_eval_count), 4),
        }


class PretrainedMuRILBaseline:
    """
    Zero-shot pretrained MuRIL baseline without contrastive fine-tuning.
    Evaluates:
    - CLS token pooling
    - Mean token pooling
    """
    def __init__(
        self,
        gazetteer: GazetteerIndex,
        pooling_mode: str = "mean",  # "cls" or "mean"
        device: torch.device = config.DEVICE
    ):
        self.gazetteer = gazetteer
        self.pooling_mode = pooling_mode
        self.device = device

        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.transformer = AutoModel.from_pretrained(config.MODEL_NAME).to(device)
        self.transformer.eval()

        self._precompute_candidates()

    def _embed_texts(self, texts: List[str]) -> torch.Tensor:
        all_embs = []
        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                out = self.transformer(**encoded)
                token_embs = out.last_hidden_state  # [B, L, H]
                mask = encoded["attention_mask"].unsqueeze(-1)  # [B, L, 1]

                if self.pooling_mode == "cls":
                    emb = token_embs[:, 0, :]
                else:  # mean pooling
                    emb = (token_embs * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

                # L2 normalize
                emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
                all_embs.append(emb.cpu())
        return torch.cat(all_embs, dim=0)

    def _precompute_candidates(self):
        self.town_names = list(self.gazetteer.town_names)
        self.town_embs = self._embed_texts(self.town_names).to(self.device)

        self.locality_town_map = []
        loc_strings = []
        for tid, loc_list in self.gazetteer.town_to_localities.items():
            tname = self.gazetteer.towns.get(tid, "")
            for lid in loc_list:
                lname = self.gazetteer.localities[lid][0]
                loc_strings.append(f"{lname}, {tname}")
                self.locality_town_map.append((lname, tname))

        self.loc_embs = self._embed_texts(loc_strings).to(self.device)

    def evaluate(self, test_df: pd.DataFrame) -> Dict[str, float]:
        town_top1 = 0
        town_top3 = 0
        town_mrr = 0.0

        loc_top1 = 0
        loc_top3 = 0
        loc_mrr = 0.0
        loc_eval_count = 0
        hier_exact = 0

        # Encode test addresses
        test_texts = [str(r["address_text"]) for _, r in test_df.iterrows()]
        addr_embs = self._embed_texts(test_texts).to(self.device)

        for i, (_, row) in enumerate(test_df.iterrows()):
            q_emb = addr_embs[i:i+1]
            true_town = str(row.get("town_name", ""))
            true_loc = str(row.get("locality_name", ""))

            # 1. Town Retrieval
            sims = torch.matmul(q_emb, self.town_embs.t()).squeeze(0).cpu().numpy()
            ranked_indices = np.argsort(-sims)
            ranked_towns = [self.town_names[idx] for idx in ranked_indices]

            if true_town in ranked_towns:
                rank = ranked_towns.index(true_town) + 1
                if rank == 1:
                    town_top1 += 1
                if rank <= 3:
                    town_top3 += 1
                town_mrr += 1.0 / rank

            pred_town = ranked_towns[0]

            # 2. Locality Retrieval
            if true_loc and true_loc != "UNKNOWN" and true_loc != "nan":
                loc_eval_count += 1
                cand_indices = [
                    idx for idx, (lname, tname) in enumerate(self.locality_town_map)
                    if tname == pred_town
                ]
                if not cand_indices:
                    cand_indices = list(range(len(self.locality_town_map)))

                sub_embs = self.loc_embs[cand_indices]
                loc_sims = torch.matmul(q_emb, sub_embs.t()).squeeze(0).cpu().numpy()
                sub_ranked = np.argsort(-loc_sims)
                ranked_locs = [self.locality_town_map[cand_indices[idx]][0] for idx in sub_ranked]

                if true_loc in ranked_locs:
                    rank = ranked_locs.index(true_loc) + 1
                    if rank == 1:
                        loc_top1 += 1
                        if pred_town == true_town:
                            hier_exact += 1
                    if rank <= 3:
                        loc_top3 += 1
                    loc_mrr += 1.0 / rank

        n_total = len(test_df)
        return {
            "town_top1": round(town_top1 / max(1, n_total), 4),
            "town_top3": round(town_top3 / max(1, n_total), 4),
            "town_mrr": round(town_mrr / max(1, n_total), 4),
            "locality_top1": round(loc_top1 / max(1, loc_eval_count), 4),
            "locality_top3": round(loc_top3 / max(1, loc_eval_count), 4),
            "locality_mrr": round(loc_mrr / max(1, loc_eval_count), 4),
            "hierarchical_exact_match": round(hier_exact / max(1, loc_eval_count), 4),
        }
