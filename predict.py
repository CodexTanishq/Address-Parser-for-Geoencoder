"""
Module: final_model/src/predict.py
Public API exposing `predict(address_text: str, address_id: str = "UNKNOWN") -> dict`
Reads messy Indian addresses and outputs:
- address_id, raw_address, town_name, locality_name, locality_x, locality_y
- landmarks: [ { landmark_name, landmark_type, relation, poi_x, poi_y, resolution_mode, candidate_count } ]
- dropped: [ { entity_name, entity_type, reason } ]
"""
from typing import Dict, Any, List, Optional
from final_model.src.fusion import AddressFusionEngine

# Singleton engine
_ENGINE: Optional[AddressFusionEngine] = None

def get_engine() -> AddressFusionEngine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = AddressFusionEngine()
    return _ENGINE

def predict(address_text: str, address_id: str = "UNKNOWN") -> Dict[str, Any]:
    """
    Main prediction entry point.
    
    Args:
        address_text: Raw messy address string.
        address_id: Optional unique identifier for the address.
        
    Returns:
        Structured dictionary matching step 5 specifications:
        {
            "address_id": str,
            "raw_address": str,
            "town_name": str,
            "locality_name": str,
            "locality_x": float or None,
            "locality_y": float or None,
            "landmarks": [
                {
                    "landmark_name": str,
                    "landmark_type": str,
                    "relation": str,
                    "poi_x": float,
                    "poi_y": float,
                    "resolution_mode": str ("exact_key", "candidates", "nearest_in_town"),
                    "candidate_count": int
                }
            ],
            "dropped": [
                {
                    "entity_name": str,
                    "entity_type": str,
                    "reason": str
                }
            ]
        }
    """
    engine = get_engine()
    res = engine.predict(address_text, address_id=address_id)
    
    return {
        "address_id": res["address_id"],
        "raw_address": res["raw_address"],
        "town_name": res["town_name"],
        "locality_name": res["locality_name"],
        "locality_x": res["locality_x"],
        "locality_y": res["locality_y"],
        "landmarks": res["landmarks"],
        "dropped": res["dropped"]
    }

if __name__ == "__main__":
    sample = "Opp. Post Office, Kuvempu Layt, 960102"
    print(f"Sample Input: {sample}")
    import json
    print(json.dumps(predict(sample, address_id="TEST001"), indent=2))
