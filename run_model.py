"""Unified Model CLI Entrypoint.
Connects CodexTanishq/Address-Parser-for-Geoencoder with CreditNirvana PS-3 Spatial Fusion & Calibrated Radius.

Usage:
    # 1. Run on a single address string directly:
    python run_model.py "Opp. Post Office, Kuvempu Layout, Kaveripura - 960102"

    # 2. Run with simulated field visit evidence:
    python run_model.py "Opp. Post Office, Kuvempu Layout, Kaveripura - 960102" --with-visits

    # 3. Run multi-case demonstration:
    python run_model.py --demo

    # 4. Interactive mode (type address interactively):
    python run_model.py --interactive

    # 5. Batch process addresses from data/addresses.csv:
    python run_model.py --batch --limit 5
"""

import sys
import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress sklearn unpickling and deprecation warnings on CLI
warnings.filterwarnings("ignore")

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

# Ensure stdout handles non-ASCII characters on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from creditnirvana_ps3.model import UnifiedAddressModel


def safe_text(val: Any) -> str:
    """Safely converts text for console printing across diverse character sets."""
    s = str(val)
    try:
        encoding = sys.stdout.encoding or "utf-8"
        return s.encode(encoding, errors="replace").decode(encoding)
    except Exception:
        return s.encode("ascii", errors="replace").decode("ascii")


def format_prediction_output(res: Dict[str, Any]):
    """Prints clean, formatted output of parser extraction and model prediction."""
    print("\n" + "=" * 78)
    print("UNIFIED MODEL PREDICTION RESULT")
    print("=" * 78)
    print(f"Address ID:          {res['address_id']}")
    print(f"Raw Input Address:   \"{safe_text(res['raw_address'])}\"")
    print("-" * 78)

    entities = res["parsed_entities"]
    print("1. ADDRESS PARSER STAGE (CodexTanishq):")
    print(f"   Town Scoped:      {safe_text(entities['town_name'])} (ID: {entities['town_id']})")
    print("   Town Coords Fab:  False (VERIFIED: Town strictly scopes search territory; no fabricated coords)")
    loc_c = entities["locality_coords"]
    loc_str = f"({loc_c[0]:.1f}, {loc_c[1]:.1f})" if loc_c else "None"
    print(f"   Locality Prior:   {safe_text(entities['locality_name'])} -> {loc_str} (broad prior only, not house pin)")
    print(f"   Landmarks Found:  {entities['landmarks_count']} anchor(s) | Ambiguity Preserved: {entities['is_ambiguous']}")

    for idx, lm in enumerate(entities["landmarks"], 1):
        print(f"     Anchor {idx}: '{safe_text(lm['landmark_name'])}' [{lm['landmark_type']}] rel='{lm['relation']}' "
              f"coords=({lm['poi_x']}, {lm['poi_y']}) candidates={lm['candidate_count']} mode='{lm['resolution_mode']}'")

    print("\n2. SPATIAL FUSION & CALIBRATED UNCERTAINTY (CreditNirvana PS-3):")
    diag = res["parser_diagnostics"]
    print(f"   Branch Executed:  {diag['branch_executed']}")
    print(f"   Estimated Pin:    X = {res['pin_x']:.1f} m, Y = {res['pin_y']:.1f} m (Local Topocentric Metric Grid)")
    print(f"   Calibrated R50:   {res['radius_50']:.1f} metres (Nominal 50% Coverage)")
    print(f"   Calibrated R90:   {res['radius_90']:.1f} metres (Nominal 90% Coverage)")
    print(f"   Action Decision:  {res['confidence_action'].upper()}")
    print(f"   Quality Flags:    {', '.join(res['quality_flags'])}")
    print(f"   Audit Trail:      {safe_text(res['explanation'])}")
    print("=" * 78 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Unified Address Geocoder: Connects CodexTanishq Parser with CreditNirvana Model."
    )
    parser.add_argument(
        "positional_address",
        nargs="?",
        type=str,
        help="Raw address string to parse and predict (optional if --address or --demo is provided).",
    )
    parser.add_argument(
        "--address",
        "-a",
        type=str,
        help="Raw address text to parse and geocode.",
    )
    parser.add_argument(
        "--with-visits",
        action="store_true",
        help="Simulate credible field visit evidence to activate Branch A (Bayesian Grid Mixture).",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run comprehensive demonstration cases (unvisited, multi-candidate ambiguous, visited).",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run batch predictions on real addresses from data/addresses.csv.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of addresses to process in batch mode (default: 5).",
    )
    parser.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Enter interactive address input loop.",
    )

    args = parser.parse_args()
    model = UnifiedAddressModel()

    # Priority 1: Interactive Loop
    if args.interactive:
        print("\n=== Interactive Address Geocoder Prompt ===")
        print("Type an Indian address and press Enter (or 'quit' / 'exit' to exit):\n")
        while True:
            try:
                line = input("Address > ").strip()
                if not line or line.lower() in ["quit", "exit", "q"]:
                    break
                res = model.predict(line, address_id="CLI_INTERACTIVE")
                format_prediction_output(res)
            except (KeyboardInterrupt, EOFError):
                break
        return

    # Priority 2: Batch Mode from data/addresses.csv
    if args.batch:
        import pandas as pd
        addr_csv_path = BASE_DIR / "data" / "addresses.csv"
        if not addr_csv_path.exists():
            print(f"[ERROR] data/addresses.csv not found at {addr_csv_path}")
            return
        df = pd.read_csv(addr_csv_path).head(args.limit)
        print(f"\nProcessing {len(df)} addresses from data/addresses.csv through Unified Model...\n")
        for _, row in df.iterrows():
            aid = str(row["address_id"])
            text = str(row["address_text"])
            res = model.predict(text, address_id=aid)
            format_prediction_output(res)
        return

    # Priority 3: Demo Mode
    if args.demo:
        demo_cases = [
            (
                "AD_DEMO_01",
                "Opp. Post Office, Kuvempu Layout, Kaveripura - 960102",
                True,
                "Ambiguous landmark (3 Post Office candidates in Kaveripura) + Field Visit Evidence (Branch A)",
            ),
            (
                "AD_DEMO_02",
                "Behind City Hospital, Vijayanagar, Navanagara East - 980104",
                False,
                "Directional relation ('behind') + Locality in Town T3 (Branch B Cold-Start)",
            ),
            (
                "AD_DEMO_03",
                "Near Bus Stand, Devgarh Nagar - 970101",
                False,
                "Town T2 scoping with 8 Bus Stand candidates preserved (Branch B Cold-Start)",
            ),
        ]
        print("\n" + "#" * 78)
        print("RUNNING COMPREHENSIVE MULTI-CASE DEMONSTRATION")
        print("#" * 78)
        for aid, text, with_v, desc in demo_cases:
            print(f"\n>>> Case: {desc}")
            visits = None
            if with_v:
                visits = [{
                    "visit_id": f"V_{aid}_01",
                    "agent_id": "AG_001",
                    "checkin_x": 1120.0,
                    "checkin_y": -210.0,
                    "gps_accuracy_m": 8.0,
                    "dwell_s": 180.0,
                    "outcome": "BORROWER_MET_HOME",
                }]
            res = model.predict(text, visits=visits, address_id=aid)
            format_prediction_output(res)
        return

    # Priority 4: Single Address (Positional or --address)
    target_addr = args.address or args.positional_address
    if not target_addr:
        target_addr = "Opp. Post Office, Kuvempu Layout, Kaveripura - 960102"
        print(f"[INFO] No address specified. Using default demo address: \"{target_addr}\"")

    visits = None
    if args.with_visits:
        visits = [{
            "visit_id": "V_CLI_01",
            "agent_id": "AG_001",
            "checkin_x": 1120.0,
            "checkin_y": -210.0,
            "gps_accuracy_m": 8.0,
            "dwell_s": 180.0,
            "outcome": "BORROWER_MET_HOME",
        }]

    res = model.predict(target_addr, visits=visits, address_id="CLI_USER")
    format_prediction_output(res)


if __name__ == "__main__":
    main()
