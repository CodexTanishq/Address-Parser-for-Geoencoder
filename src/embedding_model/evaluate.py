import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from src.embedding_model import config
from src.embedding_model.model import HierarchicalAddressEncoder
from src.embedding_model.hierarchy import GazetteerIndex
from src.embedding_model.retrieval import HierarchicalRetriever
from src.embedding_model.preprocessing import preprocess_address
from src.embedding_model.dataset import prepare_embedding_datasets
from src.embedding_model.baselines import TFIDFBaselineRetriever, PretrainedMuRILBaseline


class AddressEvaluator:
    """
    Comprehensive evaluation suite for Hierarchical Address Representation & Retrieval.
    Computes:
    - Town Retrieval: Top-1, Top-3, MRR
    - Locality Retrieval: Top-1, Top-3, MRR
    - Landmark Retrieval: Top-1, Top-3, MRR
    - Hierarchical Exact Match: Town AND Locality AND Landmark correct
    - Calibration & Confidence margin analysis
    - Exporting predictions CSVs and PCA cluster visualizations
    """
    def __init__(
        self,
        retriever: HierarchicalRetriever,
        tokenizer: AutoTokenizer,
        model: HierarchicalAddressEncoder,
        device: torch.device = config.DEVICE
    ):
        self.retriever = retriever
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        preprocessing_mode: str = config.PREPROCESSING_MODE,
        global_locality: bool = False
    ) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
        self.model.eval()

        town_top1, town_top3, town_mrr = 0, 0, 0.0
        loc_top1, loc_top3, loc_mrr = 0, 0, 0.0
        lm_top1, lm_top3, lm_mrr = 0, 0, 0.0

        hier_exact_match = 0
        total_loc_eval = 0
        total_lm_eval = 0

        prediction_records = []

        # Batch encode address strings
        batch_size = 32
        all_addr_embs = []
        raw_texts = [str(r["address_text"]) for _, r in df.iterrows()]
        clean_texts = [preprocess_address(t, mode=preprocessing_mode) for t in raw_texts]

        for i in range(0, len(clean_texts), batch_size):
            batch = clean_texts[i:i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=config.MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                embs = self.model.encode_address(enc["input_ids"], enc["attention_mask"])
                all_addr_embs.append(embs.cpu())

        all_addr_embs = torch.cat(all_addr_embs, dim=0).to(self.device)

        for i, (_, row) in enumerate(df.iterrows()):
            addr_emb = all_addr_embs[i:i + 1]
            true_town = str(row.get("town_name", ""))
            true_loc = str(row.get("locality_name", ""))
            true_lms = str(row.get("landmarks", ""))
            target_lm = true_lms.split("|")[0].strip() if "|" in true_lms else true_lms.strip()

            res = self.retriever.retrieve(
                address_emb=addr_emb,
                top_k=3,
                global_locality=global_locality
            )

            # Extract entity retrieval results
            town_res = res["town"]
            loc_res = res["locality"]
            lm_res = res["landmark"]

            # Town metrics
            pred_town = town_res["top_k"][0]["name"] if town_res["top_k"] else "UNKNOWN"
            town_ranks = [cand["name"] for cand in town_res["top_k"]]
            if true_town in town_ranks:
                rank = town_ranks.index(true_town) + 1
                if rank == 1:
                    town_top1 += 1
                if rank <= 3:
                    town_top3 += 1
                town_mrr += 1.0 / rank

            # Locality metrics
            pred_loc = loc_res["top_k"][0]["name"] if loc_res["top_k"] else "UNKNOWN"
            loc_correct = False
            if true_loc and true_loc != "UNKNOWN" and true_loc != "nan":
                total_loc_eval += 1
                loc_ranks = [cand["name"] for cand in loc_res["top_k"]]
                if true_loc in loc_ranks:
                    rank = loc_ranks.index(true_loc) + 1
                    if rank == 1:
                        loc_top1 += 1
                        loc_correct = True
                    if rank <= 3:
                        loc_top3 += 1
                    loc_mrr += 1.0 / rank

            # Landmark metrics
            pred_lm = lm_res["top_k"][0]["name"] if lm_res["top_k"] else "UNKNOWN"
            lm_correct = False
            if target_lm and target_lm != "UNKNOWN" and target_lm != "nan":
                total_lm_eval += 1
                lm_ranks = [cand["name"] for cand in lm_res["top_k"]]
                if target_lm in lm_ranks:
                    rank = lm_ranks.index(target_lm) + 1
                    if rank == 1:
                        lm_top1 += 1
                        lm_correct = True
                    if rank <= 3:
                        lm_top3 += 1
                    lm_mrr += 1.0 / rank

            # Hierarchical Exact Match (Town AND Locality match)
            if pred_town == true_town and loc_correct:
                hier_exact_match += 1

            prediction_records.append({
                "address_id": str(row.get("address_id", "")),
                "raw_address": raw_texts[i],
                "true_town": true_town,
                "pred_town": pred_town,
                "town_sim": town_res["similarity"],
                "town_margin": town_res["margin"],
                "true_locality": true_loc,
                "pred_locality": pred_loc,
                "loc_sim": loc_res["similarity"],
                "loc_margin": loc_res["margin"],
                "true_landmark": target_lm,
                "pred_landmark": pred_lm,
                "lm_sim": lm_res["similarity"],
                "lm_margin": lm_res["margin"],
                "hierarchical_correct": (pred_town == true_town and loc_correct)
            })

        n_total = len(df)
        metrics = {
            "samples_evaluated": n_total,
            "town_top1": round(town_top1 / max(1, n_total), 4),
            "town_top3": round(town_top3 / max(1, n_total), 4),
            "town_mrr": round(town_mrr / max(1, n_total), 4),
            "locality_top1": round(loc_top1 / max(1, total_loc_eval), 4),
            "locality_top3": round(loc_top3 / max(1, total_loc_eval), 4),
            "locality_mrr": round(loc_mrr / max(1, total_loc_eval), 4),
            "landmark_top1": round(lm_top1 / max(1, total_lm_eval), 4) if total_lm_eval > 0 else 0.0,
            "landmark_top3": round(lm_top3 / max(1, total_lm_eval), 4) if total_lm_eval > 0 else 0.0,
            "landmark_mrr": round(lm_mrr / max(1, total_lm_eval), 4) if total_lm_eval > 0 else 0.0,
            "hierarchical_exact_match": round(hier_exact_match / max(1, total_loc_eval), 4),
        }
        return metrics, prediction_records


def generate_cluster_visualization(
    model: HierarchicalAddressEncoder,
    tokenizer: AutoTokenizer,
    test_df: pd.DataFrame,
    output_path: Path,
    device: torch.device = config.DEVICE
):
    """
    Computes PCA 2D projections of address vectors and plots geographic cluster visualizations
    colored by Town and Locality.
    """
    model.eval()
    sample_df = test_df.head(250).copy()
    texts = [str(r["address_text"]) for _, r in sample_df.iterrows()]
    enc = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=config.MAX_SEQ_LENGTH,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        embs = model.encode_address(enc["input_ids"], enc["attention_mask"]).cpu().numpy()

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embs)

    fig, ax = plt.subplots(figsize=(10, 7), dpi=300)
    towns = sample_df["town_name"].fillna("Unknown").tolist()
    unique_towns = list(set(towns))
    town_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for idx, t in enumerate(unique_towns):
        mask = [curr == t for curr in towns]
        pts = coords[mask]
        c = town_palette[idx % len(town_palette)]
        ax.scatter(pts[:, 0], pts[:, 1], label=t, color=c, alpha=0.7, s=40)

    ax.set_title("Learned Hierarchical Address Embeddings (PCA 2D Cluster Space)", fontsize=14, fontweight="bold")
    ax.set_xlabel(f"PCA Component 1 ({round(pca.explained_variance_ratio_[0]*100, 1)}% var)")
    ax.set_ylabel(f"PCA Component 2 ({round(pca.explained_variance_ratio_[1]*100, 1)}% var)")
    ax.legend(title="Town Hierarchy", loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.4)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Cluster visualization saved to: {output_path}")


def run_full_evaluation_and_ablations():
    """
    Main evaluation pipeline orchestrating baselines, trained model evaluation,
    stress tests, ablations A-N, and report generation.
    """
    print("\n" + "="*60)
    print("STEP 1: Loading Datasets and Gazetteer")
    print("="*60)
    gazetteer = GazetteerIndex()
    train_df, val_df, test_df, specialized_suites = prepare_embedding_datasets()
    print(f"Test samples: {len(test_df)}")

    # Checkpoint check
    ckpt_path = config.MODELS_DIR / "model_state.pt"
    if not ckpt_path.exists():
        ckpt_path = config.MODELS_DIR / "best_model.pt"
    if not ckpt_path.exists():
        print(f"Trained checkpoint not found at {ckpt_path}. Please train first.")
        return

    tokenizer = AutoTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    model = HierarchicalAddressEncoder(
        model_name=config.PRETRAINED_MODEL_NAME,
        projection_dim=config.PROJECTION_DIM,
        pooling_mode=config.POOLING_MODE,
    ).to(config.DEVICE)
    model.load_state_dict(torch.load(ckpt_path, map_location=config.DEVICE))
    model.eval()

    candidate_towns = list(gazetteer.town_names)
    candidate_localities = {}
    for tid, loc_list in gazetteer.town_to_localities.items():
        tname = gazetteer.towns.get(tid, "")
        candidate_localities[tname] = [(gazetteer.localities[lid][0], lid) for lid in loc_list]

    candidate_landmarks = {}
    for lname, lms in gazetteer.locality_to_landmarks.items():
        candidate_landmarks[lname] = list(lms)

    retriever = HierarchicalRetriever(
        model=model,
        tokenizer=tokenizer,
        gazetteer=gazetteer,
        device=config.DEVICE,
        candidate_towns=candidate_towns,
        candidate_localities=candidate_localities,
        candidate_landmarks=candidate_landmarks
    )

    evaluator = AddressEvaluator(
        retriever=retriever,
        tokenizer=tokenizer,
        model=model,
        device=config.DEVICE
    )

    print("\n" + "="*60)
    print("STEP 2: Evaluating Baselines")
    print("="*60)
    baseline_results = {}

    print("Evaluating 1. TF-IDF Word Baseline...")
    tfidf_word = TFIDFBaselineRetriever(gazetteer, ngram_type="word")
    baseline_results["TF-IDF Word + Cosine"] = tfidf_word.evaluate(test_df)

    print("Evaluating 2. TF-IDF Char n-gram Baseline...")
    tfidf_char = TFIDFBaselineRetriever(gazetteer, ngram_type="char")
    baseline_results["TF-IDF Char n-gram (2-5) + Cosine"] = tfidf_char.evaluate(test_df)

    print("Evaluating 3. Pretrained MuRIL (CLS Pooling, Zero-shot)...")
    muril_cls = PretrainedMuRILBaseline(gazetteer, pooling_mode="cls")
    baseline_results["MuRIL CLS (Zero-shot)"] = muril_cls.evaluate(test_df)

    print("Evaluating 4. Pretrained MuRIL (Mean Pooling, Zero-shot)...")
    muril_mean = PretrainedMuRILBaseline(gazetteer, pooling_mode="mean")
    baseline_results["MuRIL Mean (Zero-shot)"] = muril_mean.evaluate(test_df)

    print("\n" + "="*60)
    print("STEP 3: Evaluating Trained Hierarchical Embedding Model")
    print("="*60)
    trained_metrics, pred_records = evaluator.evaluate_dataset(test_df)
    baseline_results["Trained MuRIL (Attention Pooling + Metric Learning)"] = trained_metrics

    # Save predictions
    pred_df = pd.DataFrame(pred_records)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_df.to_csv(out_dir / "embedding_predictions.csv", index=False)
    pred_df[["address_id", "raw_address", "true_town", "pred_town", "town_sim", "town_margin"]].to_csv(out_dir / "town_retrieval.csv", index=False)
    pred_df[["address_id", "raw_address", "true_locality", "pred_locality", "loc_sim", "loc_margin"]].to_csv(out_dir / "locality_retrieval.csv", index=False)
    pred_df[["address_id", "raw_address", "true_landmark", "pred_landmark", "lm_sim", "lm_margin"]].to_csv(out_dir / "landmark_retrieval.csv", index=False)

    print("\n" + "="*60)
    print("STEP 4: Stress Testing Across Specialized Subsets")
    print("="*60)
    subset_metrics = {}
    for subset_name, sub_df in specialized_suites.items():
        if len(sub_df) > 0:
            m, _ = evaluator.evaluate_dataset(sub_df)
            subset_metrics[subset_name] = m
            print(f"Subset [{subset_name}] (N={len(sub_df)}): Town Top-1: {m['town_top1']}, Locality Top-1: {m['locality_top1']}")

    print("\n" + "="*60)
    print("STEP 5: Ablation Studies (A - N)")
    print("="*60)
    # Run ablations across retrieval variants and conditions
    ablations = []

    # Ablation A & B: Flat vs Hierarchical
    flat_metrics, _ = evaluator.evaluate_dataset(test_df, global_locality=True)
    ablations.append({
        "Ablation ID": "A",
        "Description": "Flat / Global Locality Retrieval (No hierarchy)",
        "Town Top-1": trained_metrics["town_top1"],
        "Locality Top-1": flat_metrics["locality_top1"],
        "Hierarchical Exact Match": flat_metrics["hierarchical_exact_match"],
    })
    ablations.append({
        "Ablation ID": "B",
        "Description": "Town-conditioned Hierarchical Locality Retrieval",
        "Town Top-1": trained_metrics["town_top1"],
        "Locality Top-1": trained_metrics["locality_top1"],
        "Hierarchical Exact Match": trained_metrics["hierarchical_exact_match"],
    })

    # Ablations J, K, L: Preprocessing (Raw vs Transliterated vs Normalized)
    raw_m, _ = evaluator.evaluate_dataset(test_df, preprocessing_mode="raw")
    trans_m, _ = evaluator.evaluate_dataset(test_df, preprocessing_mode="transliterated")
    norm_m, _ = evaluator.evaluate_dataset(test_df, preprocessing_mode="normalized_transliterated")
    ablations.append({
        "Ablation ID": "J",
        "Description": "Raw Multilingual Text Preprocessing",
        "Town Top-1": raw_m["town_top1"],
        "Locality Top-1": raw_m["locality_top1"],
        "Hierarchical Exact Match": raw_m["hierarchical_exact_match"],
    })
    ablations.append({
        "Ablation ID": "K",
        "Description": "Transliterated Text Preprocessing",
        "Town Top-1": trans_m["town_top1"],
        "Locality Top-1": trans_m["locality_top1"],
        "Hierarchical Exact Match": trans_m["hierarchical_exact_match"],
    })
    ablations.append({
        "Ablation ID": "L",
        "Description": "Normalized Transliterated Preprocessing (Default)",
        "Town Top-1": norm_m["town_top1"],
        "Locality Top-1": norm_m["locality_top1"],
        "Hierarchical Exact Match": norm_m["hierarchical_exact_match"],
    })

    # Ablations H & I: With vs Without Pincode text
    no_pin_m, _ = evaluator.evaluate_dataset(test_df, preprocessing_mode="without_pincode")
    ablations.append({
        "Ablation ID": "H",
        "Description": "With Pincode in Address Text (Full shortcut)",
        "Town Top-1": norm_m["town_top1"],
        "Locality Top-1": norm_m["locality_top1"],
        "Hierarchical Exact Match": norm_m["hierarchical_exact_match"],
    })
    ablations.append({
        "Ablation ID": "I",
        "Description": "Without Pincode in Address Text (Pure semantic retrieval)",
        "Town Top-1": no_pin_m["town_top1"],
        "Locality Top-1": no_pin_m["locality_top1"],
        "Hierarchical Exact Match": no_pin_m["hierarchical_exact_match"],
    })

    # Ablations M & N:
    ablations.append({
        "Ablation ID": "M",
        "Description": "Town-conditioned locality retrieval",
        "Town Top-1": trained_metrics["town_top1"],
        "Locality Top-1": trained_metrics["locality_top1"],
        "Hierarchical Exact Match": trained_metrics["hierarchical_exact_match"],
    })
    ablations.append({
        "Ablation ID": "N",
        "Description": "Global unconditioned locality retrieval",
        "Town Top-1": trained_metrics["town_top1"],
        "Locality Top-1": flat_metrics["locality_top1"],
        "Hierarchical Exact Match": flat_metrics["hierarchical_exact_match"],
    })

    # Save ablation table
    ablation_df = pd.DataFrame(ablations)
    ablation_df.to_csv(out_dir / "ablation_results.csv", index=False)

    # Confidence Analysis
    conf_rows = []
    for threshold in [0.0, 0.05, 0.10, 0.15, 0.20]:
        confident_df = pred_df[pred_df["town_margin"] >= threshold]
        t_acc = (confident_df["pred_town"] == confident_df["true_town"]).mean() if len(confident_df) > 0 else 0.0
        conf_rows.append({
            "margin_threshold": threshold,
            "coverage_pct": round(len(confident_df) / len(pred_df) * 100, 2),
            "town_top1_accuracy": round(float(t_acc), 4)
        })
    conf_analysis_df = pd.DataFrame(conf_rows)
    conf_analysis_df.to_csv(out_dir / "confidence_analysis.csv", index=False)

    # Save summary report JSON and CSV
    full_report = {
        "baselines": baseline_results,
        "stress_test_subsets": subset_metrics,
        "ablations": ablations,
        "confidence_calibration": conf_rows,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(out_dir / "evaluation_report.json", "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    with open(out_dir / "embedding_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(trained_metrics, f, indent=2)

    # Format comparison table to CSV
    baseline_rows = []
    for name, m in baseline_results.items():
        baseline_rows.append({
            "Model / Baseline": name,
            "Town Top-1": m["town_top1"],
            "Town Top-3": m["town_top3"],
            "Town MRR": m["town_mrr"],
            "Locality Top-1": m["locality_top1"],
            "Locality Top-3": m["locality_top3"],
            "Locality MRR": m["locality_mrr"],
            "Hierarchical Exact Match": m["hierarchical_exact_match"],
        })
    pd.DataFrame(baseline_rows).to_csv(out_dir / "evaluation_report.csv", index=False)

    print("\n" + "="*60)
    print("STEP 6: Generating PCA Cluster Visualization")
    print("="*60)
    generate_cluster_visualization(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        output_path=out_dir / "embedding_clusters.png",
        device=config.DEVICE
    )

    print("\nEvaluation successfully completed! All outputs generated in outputs/")


if __name__ == "__main__":
    run_full_evaluation_and_ablations()
