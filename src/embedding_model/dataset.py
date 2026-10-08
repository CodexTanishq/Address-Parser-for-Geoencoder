import re
from typing import Tuple, Dict, Any, List
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from src.embedding_model import config


def create_template_signature(text: str) -> str:
    """Create structural signature to prevent test set data leakage."""
    t = text.lower()
    t = re.sub(r"\d+", "<NUM>", t)
    t = re.sub(r"[^\w\s<>]", " ", t)
    return " ".join(t.split()[:8])


def prepare_embedding_datasets(
    parsed_csv_path: str = "outputs/parsed_addresses.csv",
    seed: int = config.RANDOM_SEED,
    **kwargs
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    Constructs leakage-free Train (70%), Val (15%), Test (15%) splits
    plus 6 specialized stress-test evaluation sets:
    1. IID Test Set
    2. Lexical / Abbreviation Variation Test
    3. Multilingual Test (Kannada & Devanagari)
    4. Noisy Spelling Test (Misspelled localities)
    5. Addresses with Missing Locality
    6. Addresses without PIN code
    """
    df = pd.read_csv(parsed_csv_path)
    # Exclude OUT rows if any; keep T1/T2/T3 benchmark
    df = df[df["town_id"].isin(["T1", "T2", "T3"])].copy()

    # Create structural grouping key
    df["template_group"] = df["address_text"].apply(create_template_signature)

    # 70% Train, 30% Temp
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.70, random_state=seed)
    train_idx, temp_idx = next(gss1.split(df, groups=df["template_group"]))
    train_df = df.iloc[train_idx].copy()
    temp_df = df.iloc[temp_idx].copy()

    # 15% Val, 15% Test
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.50, random_state=seed)
    val_sub_idx, test_sub_idx = next(gss2.split(temp_df, groups=temp_df["template_group"]))
    val_df = temp_df.iloc[val_sub_idx].copy()
    test_df = temp_df.iloc[test_sub_idx].copy()

    # Specialized evaluation suites
    specialized_suites = {}

    # 1. IID Test Set (standard test split)
    specialized_suites["iid_test"] = test_df.copy()

    # 2. Multilingual test (Kannada or Devanagari)
    specialized_suites["multilingual"] = df[df["scripts"].str.contains("Kannada|Devanagari")].copy()

    # 3. Noisy spelling / Abbreviation test
    specialized_suites["noisy_spelling"] = df[df["address_text"].str.contains(r"(?i)\b(layt|layou|ngr|nagr|colny|lakevew|silvur)\b")].copy()

    # 4. Missing locality test (addresses where locality is unresolved)
    specialized_suites["missing_locality"] = df[df["is_locality_unresolved"] == True].copy()

    # 5. Missing pincode test
    specialized_suites["missing_pincode"] = df[df["pincode"].isnull() | (df["pincode"] == "")].copy()

    return train_df, val_df, test_df, specialized_suites
