"""CLI Entry Point to execute the end-to-end CreditNirvana PS3 workflow.
Usage:
    python run_pipeline.py
"""

import logging
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from creditnirvana_ps3.pipeline import CoordinatePipeline


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    setup_logging()
    logger = logging.getLogger("run_pipeline")
    logger.info("=" * 70)
    logger.info("CREDITNIRVANA PS3: LEARNED ADDRESS GEOCODER & CALIBRATED RADIUS")
    logger.info("=" * 70)

    pipeline = CoordinatePipeline()
    summary = pipeline.run()

    print("\n" + "=" * 70)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 70)
    print(f"Model Version:              {summary['model_version']}")
    print(f"Duration:                   {summary['duration_seconds']} seconds")
    print(f"Total Addresses Predicted:  {summary['total_addresses_predicted']}")
    print(f"Surveyed Truth Evaluated:   {summary['surveyed_ground_truth_count']}")
    print(f"Test Split Size:            {summary['splits_breakdown']['test_count']}")

    print("\nCANDIDATE METHODS COMPARISON (Test Set N={}):".format(summary['splits_breakdown']['test_count']))
    print("-" * 70)
    print(f"{'Method':<35} | {'Median Error':<14} | {'P90 Error':<14}")
    print("-" * 70)
    for m_name, m_res in summary["candidate_methods_comparison"].items():
        print(f"{m_name:<35} | {m_res['median_error_m']:>10.1f} m  | {m_res['p90_error_m']:>10.1f} m")

    print("\nPRIMARY METHOD TEST METRICS:")
    print("-" * 70)
    p_met = summary["test_primary_metrics"]
    print(f"Baseline Median Error:      {p_met['baseline_median_error_m']:.1f} m")
    print(f"Learned Pin Median Error:   {p_met['predicted_median_error_m']:.1f} m")
    print(f"Median Improvement:         {p_met['median_improvement_m']:.1f} m")
    print(f"Measured R50 Coverage:      {p_met['measured_r50_coverage_pct']:.1f}% (Nominal: 50.0%)")
    print(f"Measured R90 Coverage:      {p_met['measured_r90_coverage_pct']:.1f}% (Nominal: 90.0%)")
    print(f"Median R90 Radius Size:     {p_met['median_radius_90_m']:.1f} m")

    print("\nSPECIAL ANALYSES:")
    print("-" * 70)
    sa = summary["special_analyses"]
    print(f"Cold Start Median Error:    {sa['cold_start_error_m']:.1f} m (vs {sa['with_visits_error_m']:.1f} m with visits)")
    print(f"Adversarial Poisoning Shift:{sa['adversarial_poisoning_shift_m']:.1f} m (vs {sa['naive_poisoning_shift_m']:.1f} m naive averaging)")
    print(f"Spatial Leakage Audit:      {sa['spatial_leakage_status']}")

    print("\nGENERATED OUTPUTS & REPORTS:")
    print("-" * 70)
    for name, path in summary["output_files"].items():
        print(f"- {name:<26}: {path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
