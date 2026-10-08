import sys
import io
import re
import pandas as pd
import numpy as np

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("="*60)
print("STEP 1: THOROUGH DATA INSPECTION")
print("="*60)

addresses_df = pd.read_csv('data/addresses.csv')
towns_df = pd.read_csv('data/towns.csv')
localities_df = pd.read_csv('data/localities.csv')

# 1 & 2. Schemas and Data Types
print("\n--- 1 & 2. Schemas and Data Types ---")
for name, df in [("addresses", addresses_df), ("towns", towns_df), ("localities", localities_df)]:
    print(f"\n{name} ({len(df)} rows):")
    for col, dtype in df.dtypes.items():
        print(f"  {col}: {dtype}")

# 3. Missing Values
print("\n--- 3. Missing Values ---")
for name, df in [("addresses", addresses_df), ("towns", towns_df), ("localities", localities_df)]:
    nulls = df.isnull().sum()
    print(f"{name} missing values:\n{nulls[nulls > 0] if (nulls > 0).any() else '  None'}")

# 4 & 5. Duplicate IDs and Duplicate Addresses
print("\n--- 4 & 5. Duplicate IDs and Addresses ---")
print("Duplicate address_id count:", addresses_df['address_id'].duplicated().sum())
print("Duplicate address_text count:", addresses_df['address_text'].duplicated().sum())
if addresses_df['address_text'].duplicated().sum() > 0:
    dups = addresses_df[addresses_df['address_text'].duplicated(keep=False)].sort_values('address_text')
    print("Sample duplicate addresses:")
    for _, row in dups.head(6).iterrows():
        print(f"  [{row.address_id} - {row.town_id}]: {row.address_text}")

# 6. Representative Addresses
print("\n--- 6. Representative Addresses by Town ---")
for tid, group in addresses_df.groupby('town_id'):
    print(f"\nTown {tid} (Total: {len(group)}):")
    for _, row in group.head(3).iterrows():
        print(f"  [{row.address_id}] {row.address_text}")

# 7, 8, 9. Scripts (Kannada, Devanagari, Mixed)
print("\n--- 7, 8, 9. Scripts Analysis ---")
def get_scripts(text):
    scripts = set()
    for ch in text:
        cp = ord(ch)
        if 0x0900 <= cp <= 0x097F:
            scripts.add('Devanagari')
        elif 0x0C80 <= cp <= 0x0CFF:
            scripts.add('Kannada')
        elif 'a' <= ch.lower() <= 'z':
            scripts.add('Latin')
    return sorted(list(scripts))

addresses_df['scripts'] = addresses_df['address_text'].apply(lambda x: '+'.join(get_scripts(x)))
print("Script counts:")
print(addresses_df['scripts'].value_counts())

# 10. Abbreviations
print("\n--- 10. Inspection of Common Abbreviations ---")
abbrev_patterns = [
    r'\bno\b', r'\bno\.', r'\bh\.?no\b', r'\bhno\b', r'\bflat\b', r'\bplot\b', r'\bblk\b',
    r'\brd\b', r'\brd\.', r'\bcr\b', r'\bcrs\b', r'\bmn\b', r'\bln\b', r'\bclny\b', r'\bcolny\b',
    r'\bngr\b', r'\bnagar\b', r'\blayt\b', r'\blayou\b', r'\bnr\b', r'\bnr\.', r'\bopp\b', r'\bopp\.'
]
for p in abbrev_patterns:
    cnt = addresses_df['address_text'].str.contains(p, case=False, regex=True).sum()
    print(f"  Pattern '{p}': {cnt} occurrences")

# 11. Spelling Mistakes & Variants
print("\n--- 11. Spelling Mistakes & Variants in Localities ---")
known_loc_words = ['nagar', 'layout', 'colony', 'badavane', 'enclave', 'gardens', 'meadows', 'phase', 'basti', 'mohalla', 'puri']
misspellings = ['ngr', 'nagr', 'layt', 'layou', 'colny', 'clny', 'bdvne', 'badavne', 'enclav', 'grdns', 'mdws', 'phse', 'bsti', 'mhlla']
for m in misspellings:
    cnt = addresses_df['address_text'].str.contains(rf'\b{m}\b', case=False, regex=True).sum()
    if cnt > 0:
        print(f"  Misspelling '{m}': {cnt} matches")

# 12. Multiple Numbers
print("\n--- 12. Addresses with Multiple Numbers ---")
number_pattern = re.compile(r'\b\d+(?:[/-]\d+)?\b')
def count_numbers(text):
    return len(number_pattern.findall(text))
addresses_df['num_numbers'] = addresses_df['address_text'].apply(count_numbers)
print("Distribution of number count per address:")
print(addresses_df['num_numbers'].value_counts().sort_index())

# 13. Pincode Extraction Inspection
print("\n--- 13. Pincode Inspection ---")
pincode_regex = re.compile(r'(?<!\d)(\d{6})(?!\d)')
def extract_pincode(text):
    matches = pincode_regex.findall(text)
    return matches

addresses_df['pincodes'] = addresses_df['address_text'].apply(extract_pincode)
addresses_df['has_pincode'] = addresses_df['pincodes'].apply(lambda x: len(x) > 0)
addresses_df['pincode_val'] = addresses_df['pincodes'].apply(lambda x: x[0] if len(x) == 1 else ('MULTIPLE' if len(x) > 1 else 'MISSING'))

print("Pincode presence by town:")
for tid, group in addresses_df.groupby('town_id'):
    missing_cnt = (group['pincode_val'] == 'MISSING').sum()
    multi_cnt = (group['pincode_val'] == 'MULTIPLE').sum()
    single_cnt = len(group) - missing_cnt - multi_cnt
    print(f"  Town {tid}: Single={single_cnt}, Missing={missing_cnt}, Multiple={multi_cnt}")

# 14 & 15. Locality Matching and Pincode vs Locality Agreement
print("\n--- 14 & 15. Locality Matching & Pincode-Locality Agreement ---")
# Build lookup of locality pincodes
loc_pin_map = {}
for _, row in localities_df.iterrows():
    loc_pin_map[row.locality_name.lower()] = (row.town_id, str(row.pincode))

print("Gazetteer localities count:", len(localities_df))
print(f"Unique gazetteer pincodes: {localities_df['pincode'].nunique()}")
print("Gazetteer pincodes per town:")
for tid, group in localities_df.groupby('town_id'):
    print(f"  Town {tid}: {group['pincode'].unique().tolist()} ({len(group)} localities)")
