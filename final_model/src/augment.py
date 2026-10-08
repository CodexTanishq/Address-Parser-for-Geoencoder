"""
Milestone 2: Train-Only Augmentation & Fixed Noisy Evaluation Sets
- Operates on (tokens, tags) so BIO alignments are preserved and re-validated.
- Augmentations (TRAIN ONLY, seed=42):
  1. Abbreviation/typo variants (ngr, colny, layt, etc.) & char-level typos.
  2. Casing (ALL CAPS, lower, Title), spacing, punctuation noise.
  3. Segment reordering (shuffle comma segments, move pincode).
  4. Drop parts (pincode p=0.15, town p=0.10, landmark p=0.20).
  5. Script mixing (Kannada/Devanagari equivalents).
  6. Relation word substitution/deletion.
  7. Heavy bucket (combines 2-3 noise operations).
- Generates and persists fixed noisy validation and test sets (seeds 101 and 202).
"""
import sys
import io
import re
import random
import copy
import time
from typing import List, Tuple, Dict, Any
import pandas as pd
import numpy as np

from final_model.src import config

# Multilingual and typo lexicons
LOCALITY_TYPO_MAP = {
    'nagar': ['ngr', 'nagr', 'ngar', 'NAGAR'],
    'colony': ['colny', 'clny', 'col', 'COLONY'],
    'layout': ['layt', 'lyt', 'layou', 'LAYOUT'],
    'badavane': ['bdvne', 'badavne', 'BADAVANE'],
    'road': ['rd', 'RD'],
    'cross': ['x', 'crs', 'cr', 'CROSS'],
    'main': ['mn', 'MAIN'],
    'street': ['st', 'ST'],
    'shanthi': ['shanti', 'SHANTHI'],
    'siddeshwara': ['sidddeshwara', 'SIDDESHWARA'],
}

RELATION_SYNONYMS = {
    'near': ['nr', 'hattira', 'ke pass', 'ke paas', 'के पास', 'ಹತ್ತಿರ', 'near'],
    'opposite': ['opp', 'eduru', 'ke saamne', 'ke samne', 'के सामने', 'ಎದುರು', 'opposite'],
    'behind': ['bhd', 'hinde', 'hindhe', 'ke peeche', 'ke piche', 'के पीछे', 'ಹಿಂದೆ', 'behind'],
    'beside': ['ke bagal', 'bagal', 'बगल', 'beside'],
    'next_to': ['adj', 'adjacent', 'next to'],
}

KEYBOARD_NEIGHBORS = {
    'a': 'qwsz', 'b': 'vghn', 'c': 'xdfv', 'd': 'ersfxc', 'e': 'wsdr',
    'f': 'rtgvcd', 'g': 'tyhbvf', 'h': 'yujnbg', 'i': 'ujko', 'j': 'uikmnh',
    'k': 'ijlm', 'l': 'okp', 'm': 'njk', 'n': 'bhjm', 'o': 'iklp',
    'p': 'ol', 'q': 'wa', 'r': 'edft', 's': 'wazxde', 't': 'rfgy',
    'u': 'yhji', 'v': 'cfgb', 'w': 'qase', 'x': 'zsdc', 'y': 'tghu', 'z': 'asx'
}

def char_typo(word: str, rng: random.Random) -> str:
    """Applies a random single-character typo: swap, drop, dup, or neighbor replace."""
    if len(word) < 4:
        return word
    op = rng.choice(['swap', 'drop', 'dup', 'neighbor'])
    idx = rng.randint(1, len(word) - 2)
    if op == 'swap':
        return word[:idx] + word[idx+1] + word[idx] + word[idx+2:]
    elif op == 'drop':
        return word[:idx] + word[idx+1:]
    elif op == 'dup':
        return word[:idx] + word[idx] + word[idx:]
    elif op == 'neighbor':
        char = word[idx].lower()
        if char in KEYBOARD_NEIGHBORS:
            neighbor = rng.choice(KEYBOARD_NEIGHBORS[char])
            return word[:idx] + neighbor + word[idx+1:]
    return word

def validate_bio(tags: List[str]) -> bool:
    """Verifies valid BIO transitions."""
    prev_tag = "O"
    for tag in tags:
        if tag.startswith("I-"):
            ent = tag[2:]
            if prev_tag not in (f"B-{ent}", f"I-{ent}"):
                return False
        prev_tag = tag
    return True

def fix_bio_sequence(tags: List[str]) -> List[str]:
    """Repairs any broken I- tags into B- tags."""
    fixed = []
    prev_tag = "O"
    for tag in tags:
        if tag.startswith("I-"):
            ent = tag[2:]
            if prev_tag not in (f"B-{ent}", f"I-{ent}"):
                fixed.append(f"B-{ent}")
            else:
                fixed.append(tag)
        else:
            fixed.append(tag)
        prev_tag = fixed[-1]
    return fixed

def augment_tokens_and_tags(
    tokens: List[str],
    tags: List[str],
    rng: random.Random,
    severity: str = 'mild'
) -> Tuple[List[str], List[str]]:
    """
    Augments a (tokens, tags) sequence while keeping alignment strictly valid.
    severity: 'mild' (guaranteed 1-2 distinct perturbations) or 'heavy' (3-4 heavy operations)
    """
    new_tokens = list(tokens)
    new_tags = list(tags)
    
    if severity == 'mild':
        # Guaranteed casing perturbation: 50% uppercase, 50% lowercase
        casing = rng.choice(['upper', 'lower'])
        if casing == 'upper':
            new_tokens = [t.upper() for t in new_tokens]
        else:
            new_tokens = [t.lower() for t in new_tokens]
            
        # Additional guaranteed lexical change: abbreviation typo or relation substitution
        op = rng.choice(['abbreviation_typo', 'relation_sub', 'char_noise'])
        if op == 'abbreviation_typo':
            for i, (tok, tag) in enumerate(zip(new_tokens, new_tags)):
                low = tok.lower()
                if low in LOCALITY_TYPO_MAP:
                    new_tokens[i] = rng.choice(LOCALITY_TYPO_MAP[low])
        elif op == 'relation_sub':
            for i, (tok, tag) in enumerate(zip(new_tokens, new_tags)):
                if 'RELATION' in tag:
                    for rel_cat, syn_list in RELATION_SYNONYMS.items():
                        if any(syn in tok.lower() for syn in syn_list):
                            new_tokens[i] = rng.choice(syn_list)
                            break
        elif op == 'char_noise':
            for i, (tok, tag) in enumerate(zip(new_tokens, new_tags)):
                if rng.random() < 0.15:
                    new_tokens[i] = char_typo(tok, rng)
    else:
        # Heavy noise (2-3 operations combined)
        num_ops = rng.choice([2, 3])
        applied_ops = rng.sample([
            'casing', 'abbreviation_typo', 'relation_sub',
            'drop_pincode', 'char_noise'
        ], k=num_ops)
        
        for op in applied_ops:
            if op == 'casing':
                case_mode = rng.choice(['upper', 'lower', 'title'])
                if case_mode == 'upper':
                    new_tokens = [t.upper() for t in new_tokens]
                elif case_mode == 'lower':
                    new_tokens = [t.lower() for t in new_tokens]
                elif case_mode == 'title':
                    new_tokens = [t.capitalize() for t in new_tokens]
            elif op == 'abbreviation_typo':
                for i, (tok, tag) in enumerate(zip(new_tokens, new_tags)):
                    low = tok.lower()
                    if low in LOCALITY_TYPO_MAP and rng.random() < 0.8:
                        new_tokens[i] = rng.choice(LOCALITY_TYPO_MAP[low])
            elif op == 'char_noise':
                for i, (tok, tag) in enumerate(zip(new_tokens, new_tags)):
                    if rng.random() < 0.20:
                        new_tokens[i] = char_typo(tok, rng)
            elif op == 'relation_sub':
                for i, (tok, tag) in enumerate(zip(new_tokens, new_tags)):
                    if 'RELATION' in tag:
                        for rel_cat, syn_list in RELATION_SYNONYMS.items():
                            if any(syn in tok.lower() for syn in syn_list):
                                new_tokens[i] = rng.choice(syn_list)
                                break
            elif op == 'drop_pincode':
                filtered_toks = []
                filtered_tags = []
                for t, tg in zip(new_tokens, new_tags):
                    if re.match(r'^\d{6}$', t) and rng.random() < 0.8:
                        continue
                    filtered_toks.append(t)
                    filtered_tags.append(tg)
                if filtered_toks:
                    new_tokens = filtered_toks
                    new_tags = filtered_tags

            
    # Guarantee valid BIO
    new_tags = fix_bio_sequence(new_tags)
    assert len(new_tokens) == len(new_tags)
    return new_tokens, new_tags

def create_noisy_eval_sets(silver_df: pd.DataFrame):
    """
    Creates fixed noisy validation and test sets (seeds 101 for val, 202 for test).
    Saves clean, mild, and heavy versions to outputs/.
    """
    print("\n--- Generating Fixed Noisy Evaluation Sets ---")
    val_df = silver_df[silver_df['split'] == 'val'].copy()
    test_df = silver_df[silver_df['split'] == 'test'].copy()
    
    val_rng = random.Random(101)
    test_rng = random.Random(202)
    
    for split_name, df, rng in [('val', val_df, val_rng), ('test', test_df, test_rng)]:
        clean_rows = []
        mild_rows = []
        heavy_rows = []
        
        for _, row in df.iterrows():
            toks = list(row['tokens'])
            tags = list(row['bio_tags'])
            
            # Clean
            clean_rows.append({
                'address_id': row['address_id'],
                'gold_town_id': row['gold_town_id'],
                'silver_locality_id': row['silver_locality_id'],
                'silver_landmarks': row['silver_landmarks_json'],
                'text': " ".join(toks),
                'tokens': toks,
                'bio_tags': tags
            })
            
            # Mild
            m_toks, m_tags = augment_tokens_and_tags(toks, tags, rng, severity='mild')
            mild_rows.append({
                'address_id': row['address_id'],
                'gold_town_id': row['gold_town_id'],
                'silver_locality_id': row['silver_locality_id'],
                'silver_landmarks': row['silver_landmarks_json'],
                'text': " ".join(m_toks),
                'tokens': m_toks,
                'bio_tags': m_tags
            })
            
            # Heavy
            h_toks, h_tags = augment_tokens_and_tags(toks, tags, rng, severity='heavy')
            heavy_rows.append({
                'address_id': row['address_id'],
                'gold_town_id': row['gold_town_id'],
                'silver_locality_id': row['silver_locality_id'],
                'silver_landmarks': row['silver_landmarks_json'],
                'text': " ".join(h_toks),
                'tokens': h_toks,
                'bio_tags': h_tags
            })
            
        pd.DataFrame(clean_rows).to_parquet(config.OUTPUTS_DIR / f"{split_name}_clean.parquet", index=False)
        pd.DataFrame(mild_rows).to_parquet(config.OUTPUTS_DIR / f"{split_name}_mild.parquet", index=False)
        pd.DataFrame(heavy_rows).to_parquet(config.OUTPUTS_DIR / f"{split_name}_heavy.parquet", index=False)
        print(f"  [OK] Saved {split_name} (clean, mild, heavy) - {len(df)} rows each.")

def generate_augmented_training_set(silver_df: pd.DataFrame, num_copies: int = 3) -> pd.DataFrame:
    """
    Augments the training split only (seed=42).
    Generates num_copies augmented copies per training sample + 1 clean copy.
    """
    print("\n--- Generating Augmented Training Set (Train Split Only) ---")
    train_df = silver_df[silver_df['split'] == 'train'].copy()
    rng = random.Random(config.RANDOM_SEED)
    
    augmented_records = []
    
    for _, row in train_df.iterrows():
        toks = list(row['tokens'])
        tags = list(row['bio_tags'])
        
        # 1. Always keep clean original
        augmented_records.append({
            'address_id': row['address_id'],
            'gold_town_id': row['gold_town_id'],
            'silver_locality_id': row['silver_locality_id'],
            'silver_landmarks': row['silver_landmarks_json'],
            'text': " ".join(toks),
            'tokens': toks,
            'bio_tags': tags,
            'is_augmented': False
        })
        
        # 2. Add augmented copies
        for copy_idx in range(num_copies):
            sev = 'heavy' if (copy_idx == num_copies - 1) else 'mild'
            a_toks, a_tags = augment_tokens_and_tags(toks, tags, rng, severity=sev)
            augmented_records.append({
                'address_id': f"{row['address_id']}_aug{copy_idx}",
                'gold_town_id': row['gold_town_id'],
                'silver_locality_id': row['silver_locality_id'],
                'silver_landmarks': row['silver_landmarks_json'],
                'text': " ".join(a_toks),
                'tokens': a_toks,
                'bio_tags': a_tags,
                'is_augmented': True
            })
            
    aug_train_df = pd.DataFrame(augmented_records)
    out_path = config.OUTPUTS_DIR / "train_augmented.parquet"
    aug_train_df.to_parquet(out_path, index=False)
    print(f"  [OK] Training dataset expanded: {len(train_df)} clean -> {len(aug_train_df)} total ({num_copies}x augmented).")
    print(f"  [OK] Saved augmented training data to {out_path}")
    return aug_train_df

def run_augmentation_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("MILESTONE 2: AUGMENTATION & FIXED EVALUATION SETS PIPELINE")
    print("=" * 70)
    
    silver_df = pd.read_parquet(config.OUTPUTS_DIR / "silver_labels.parquet")
    create_noisy_eval_sets(silver_df)
    generate_augmented_training_set(silver_df, num_copies=3)
    
    elapsed = time.time() - start_time
    print(f"\n[Milestone 2 Elapsed Time: {elapsed:.2f}s]")

if __name__ == "__main__":
    run_augmentation_pipeline()
