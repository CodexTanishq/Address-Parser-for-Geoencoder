import json
from pathlib import Path
from src import config
from src.evaluate_embedding import run_test_evaluation
from src.embedding_inference import AddressEmbeddingPredictor


def main():
    print("=" * 80)
    print("  VERIFICATION PIPELINE: DUAL-ENCODER ADDRESS EMBEDDING MODEL V1")
    print("=" * 80)

    # 1. Run strict test split evaluation
    metrics = run_test_evaluation()

    # 2. Test out-of-order & noisy addresses using AddressEmbeddingPredictor
    print("=" * 80)
    print("  DEMONSTRATING VECTOR LOOKUP ON PERMUTED & NOISY ADDRESSES")
    print("=" * 80)

    predictor = AddressEmbeddingPredictor()

    test_queries = [
        # Query 1: Reverse order (Pincode & City first, then typo in Locality)
        "560001 Kaveripura, Kuvempu Layt, #12, 4th Cross",
        # Query 2: Shuffled fields + Noisy delimiters
        "Opp. Post Office / Gandhi Nagr / 2nd Main / 960102 Kaveripura",
        # Query 3: Multi-script phonetics
        "No. 45, ಶಾಂತಿ ನಗರ, Kaveripura - 960103",
        # Query 4: Unseen noisy locality in Navanagara East
        "Subhash Marg, 201 Sunrise Apts, Navanagara East 560066",
    ]

    for q in test_queries:
        print(f"\n[QUERY]: \"{q}\"")
        res = predictor.retrieve(q, top_k=3)
        top1 = res["top_1_prediction"]
        print(f"  -> Top-1 Predicted  : {top1['canonical_name']} (ID: {top1['locality_id']})")
        print(f"  -> Cosine Sim Score : {top1['confidence_score']:.4f} | Margin: {top1['margin_to_next']:.4f}")
        print("  -> Top Candidates   :")
        for cand in res["top_k_candidates"]:
            print(f"      [{cand['rank']}] {cand['canonical_name']:30s} (Sim: {cand['cosine_similarity']:.4f})")

    print("\n" + "=" * 80)
    print("VERIFICATION COMPLETED SUCCESSFULLY!")
    print(f"Model Checkpoint : {config.ADDRESS_EMBEDDING_MODEL_V1_DIR}")
    print(f"Metrics Output   : {config.EMBEDDING_METRICS_JSON}")
    print("=" * 80)


if __name__ == "__main__":
    main()
