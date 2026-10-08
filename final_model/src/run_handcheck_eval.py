"""
Evaluation Script for 30 Hand-Checked Validation Addresses.
Compares model predictions against handcheck_30_gold.csv.
Computes accuracy for:
- Town (N=30)
- Locality (N=30)
- Landmark (N=30)
Generates final_model/outputs/handcheck_30_result.md.
"""
import io
import sys
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from final_model.src.predict import predict

def run_handcheck_eval():
    gold_df = pd.read_csv('final_model/outputs/handcheck_30_gold.csv')
    
    results = []
    town_correct = 0
    loc_correct = 0
    lm_correct = 0
    n_total = len(gold_df)

    print(f"Running evaluation on {n_total} hand-checked addresses...\n")

    for i, row in gold_df.iterrows():
        aid = row['address_id']
        raw = row['raw_address']
        exp_town = row['expected_town_name']
        exp_loc = row['expected_locality_name']
        exp_lm_type = row['expected_landmark_type']
        exp_lm_name = row['expected_landmark_name']

        pred = predict(raw, address_id=aid)

        pred_town = pred['town_name']
        pred_loc = pred['locality_name']
        pred_lms = pred['landmarks']

        # Town match
        t_ok = (pred_town == exp_town)
        if t_ok:
            town_correct += 1

        # Locality match
        l_ok = (pred_loc == exp_loc)
        if l_ok:
            loc_correct += 1

        # Landmark match
        pred_lm_types = [lm['landmark_type'] for lm in pred_lms]
        if exp_lm_type == 'UNKNOWN':
            lm_ok = (len(pred_lms) == 0)
        else:
            lm_ok = (exp_lm_type in pred_lm_types)
        
        if lm_ok:
            lm_correct += 1

        status_str = f"Town: {'PASS' if t_ok else 'FAIL'} | Loc: {'PASS' if l_ok else 'FAIL'} | LM: {'PASS' if lm_ok else 'FAIL'}"
        print(f"[{i+1:2d}/30] {aid} -> {status_str}")
        if not (t_ok and l_ok and lm_ok):
            print(f"       Raw: {raw}")
            print(f"       Exp : Town={exp_town}, Loc={exp_loc}, LM={exp_lm_type}")
            print(f"       Pred: Town={pred_town}, Loc={pred_loc}, LMs={pred_lm_types}")

        results.append({
            'address_id': aid,
            'raw_address': raw,
            'expected_town': exp_town,
            'pred_town': pred_town,
            'town_ok': t_ok,
            'expected_locality': exp_loc,
            'pred_locality': pred_loc,
            'loc_ok': l_ok,
            'expected_landmark_type': exp_lm_type,
            'pred_landmark_types': pred_lm_types,
            'pred_modes': [lm['resolution_mode'] for lm in pred_lms],
            'lm_ok': lm_ok
        })

    town_acc = town_correct / n_total
    loc_acc = loc_correct / n_total
    lm_acc = lm_correct / n_total

    print("\n" + "=" * 50)
    print("HAND-CHECK 30 EVALUATION SUMMARY:")
    print(f"  Town Accuracy     : {town_correct}/{n_total} ({town_acc*100:.1f}%)")
    print(f"  Locality Accuracy : {loc_correct}/{n_total} ({loc_acc*100:.1f}%)")
    print(f"  Landmark Accuracy : {lm_correct}/{n_total} ({lm_acc*100:.1f}%)")
    print("=" * 50)

    # Write handcheck_30_result.md
    md_lines = [
        "# Hand-Checked Validation Set Results (N=30)\n",
        "## Summary Metrics",
        f"- **Total Addresses (N)**: {n_total}",
        f"- **Town Accuracy**: **{town_correct}/{n_total} ({town_acc*100:.1f}%)** (Threshold: $\\ge 95\\%$ -> {'PASS' if town_acc >= 0.95 else 'FAIL'})",
        f"- **Locality Accuracy**: **{loc_correct}/{n_total} ({loc_acc*100:.1f}%)** (Threshold: $\\ge 85\\%$ -> {'PASS' if loc_acc >= 0.85 else 'FAIL'})",
        f"- **Landmark Accuracy**: **{lm_correct}/{n_total} ({lm_acc*100:.1f}%)** (Threshold: $\\ge 85\\%$ -> {'PASS' if lm_acc >= 0.85 else 'FAIL'})\n",
        "## Detailed Address-by-Address Evaluation Table",
        "| # | ID | Raw Address | Expected Town | Pred Town | Expected Locality | Pred Locality | Expected LM | Pred LM | Status |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |"
    ]

    for i, r in enumerate(results):
        status = "ALL PASS" if (r['town_ok'] and r['loc_ok'] and r['lm_ok']) else "MISMATCH"
        pred_lms_str = ", ".join(r['pred_landmark_types']) if r['pred_landmark_types'] else "NONE"
        md_lines.append(
            f"| {i+1} | {r['address_id']} | `{r['raw_address'][:45]}...` | {r['expected_town']} | {r['pred_town']} | {r['expected_locality']} | {r['pred_locality']} | {r['expected_landmark_type']} | {pred_lms_str} | **{status}** |"
        )

    with open('final_model/outputs/handcheck_30_result.md', 'w', encoding='utf-8') as f:
        f.write("\n".join(md_lines) + "\n")

    print("\nSaved report to final_model/outputs/handcheck_30_result.md")
    return town_acc, loc_acc, lm_acc

if __name__ == '__main__':
    run_handcheck_eval()
