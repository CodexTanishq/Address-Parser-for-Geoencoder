"""
List landmark false positives on the test split (read-only).
FP = a landmark TYPE the model returned that the labeller did not.
Output: outputs/landmark_false_positives_test.csv
"""
import io
import sys
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from final_model.src import config
from final_model.src.fusion import AddressFusionEngine


def run():
    engine = AddressFusionEngine()
    test_df = pd.read_parquet(config.OUTPUTS_DIR / "test_clean.parquet")

    rows = []
    for _, r in test_df.iterrows():
        p = engine.predict(r['text'], address_id=r['address_id'], use_pincode=True)

        gold_types = {lm['landmark_type'] for lm in r['silver_landmarks']}

        pred_groups = {}
        for lm in p['landmarks']:
            pred_groups.setdefault(lm['landmark_type'], []).append(lm)

        for t, pois in pred_groups.items():
            if t not in gold_types:
                rows.append({
                    'address_id': r['address_id'],
                    'text': r['text'],
                    'extra_type': t,
                    'labeller_types': ", ".join(sorted(gold_types)) or "(none)",
                    'model_locality': p['locality_id'],
                    'mode': pois[0]['resolution_mode'],
                    'candidate_count': pois[0].get('candidate_count'),
                })

    out_df = pd.DataFrame(rows)
    print(f"Total false positives: {len(out_df)}\n")
    for i, row in out_df.iterrows():
        print(f"{i+1}. {row['address_id']} | extra type: {row['extra_type']} "
              f"| labeller: {row['labeller_types']}")
        print(f"   {row['text']}")
        print(f"   locality: {row['model_locality']} | mode: {row['mode']} "
              f"| candidates: {row['candidate_count']}\n")

    out = config.OUTPUTS_DIR / "landmark_false_positives_test.csv"
    out_df.to_csv(out, index=False, encoding="utf-8-sig")
    print("Saved:", out)


if __name__ == '__main__':
    run()