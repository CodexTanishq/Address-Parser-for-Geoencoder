# Weak-Supervision Indian Address NER Pipeline

A high-precision weak-supervision and Transformer NER pipeline specifically engineered for unstructured, multi-script, and messy Indian addresses.

---

## 1. Project Philosophy & Core Objective

In messy Indian address parsing, heuristic rule-based systems hit diminishing returns due to unstructured syntax, colloquial variations, missing tokens, and transliteration noise.
However, training a deep neural model requires rich supervision:

$$\text{Deterministic Parser (High-Precision Teacher)} \xrightarrow{\text{Weak Labels}} \text{MuRIL Transformer NER (Generalizer)}$$

- **The Deterministic Parser is the Teacher**: Prioritizes **PRECISION over RECALL**. Never guesses when evidence is weak. Ambiguous or low-confidence spans are left unlabelled.
- **The Transformer NER Model is the Generalizer**: Trained on strictly high-confidence silver labels to learn contextual syntax and generalize to unseen localities, informal abbreviations, and novel layouts.

---

## 2. Supported Entity Classes

| Entity | Description | BIO Tags |
| :--- | :--- | :--- |
| `HOUSE_NUMBER` | Flat, plot, door, slash, or prefixed house identifier | `B-HOUSE_NUMBER`, `I-HOUSE_NUMBER` |
| `STREET` | Cross, Main, Road, Gali, Lane, and street ordinals | `B-STREET`, `I-STREET` |
| `WARD` | Municipal administrative ward designators | `B-WARD`, `I-WARD` |
| `LANDMARK` | Points of interest, temples, schools, tanks, relational cues | `B-LANDMARK`, `I-LANDMARK` |
| `LOCALITY` | Neighborhood, layout, enclave, or colony | `B-LOCALITY`, `I-LOCALITY` |
| `TOWN` | Authoritative city or municipal town | `B-TOWN`, `I-TOWN` |
| `PINCODE` | Exactly six-digit postal code | `B-PINCODE`, `I-PINCODE` |
| `O` | Outside / unallocated tokens / punctuation | `O` |

---

## 3. Modular Architecture

```
address_parser/
├── data/
│   ├── addresses.csv           # Immutable raw address inputs
│   ├── towns.csv               # Authoritative town gazetteer
│   └── localities.csv          # Town-scoped locality gazetteer
├── src/
│   ├── config.py               # Paths, entity classes, BIO labels, thresholds
│   ├── transliteration.py      # Indic scripts (Kannada, Devanagari, etc.) with span mapping
│   ├── normalization.py        # Text & abbreviation normalization
│   ├── pincode.py              # 6-digit pincode extraction & town verification
│   ├── town.py                 # Authoritative town resolver & span finder
│   ├── locality.py             # Gazetteer locality resolver with fuzzy & ambiguity checks
│   ├── house_number.py         # Robust house number extraction (prefixes, slashes, plots)
│   ├── street.py               # Indian street patterns (cross, main, gali, rd)
│   ├── ward.py                 # Ward extraction
│   ├── landmark.py             # POIs & multilingual contextual indicators (nr, opp, hattira, ke paas)
│   ├── parser.py               # Master coordinator, conflict resolution & other_details
│   ├── weak_labels.py          # Weak span & token label dataset generator
│   ├── bio.py                  # Tokenization, BIO label alignment & transition validator
│   ├── quality.py              # Quality reporting, metrics & human inspection sampler
│   ├── main.py                 # Master execution pipeline
│   ├── train_ner.py            # MuRIL fine-tuning pipeline with leakage-free group split
│   └── inference.py            # End-to-end inference CLI and Python API
├── tests/
│   ├── test_pincode.py         # Unit tests for pincode boundaries and ambiguity
│   ├── test_house_number.py    # Unit tests for house numbers and ordinal rejection
│   ├── test_locality.py        # Unit tests for locality matching and town scoping
│   ├── test_street.py          # Unit tests for street patterns and separate entities
│   ├── test_landmark.py        # Unit tests for landmark cues and protection
│   └── test_bio.py             # Unit tests for token alignment and BIO transitions
└── outputs/
    ├── parsed_addresses.csv    # Parsed structured addresses with overall confidence
    ├── weak_labels.csv         # Token-level weak supervision dataset
    ├── weak_spans.csv          # Span-level metadata with extraction source
    ├── bio_dataset.csv         # Sequence-level BIO dataset for Transformer NER
    ├── unresolved_addresses.csv# Addresses with unresolved localities for manual curation
    ├── ambiguous_addresses.csv # Ambiguous matches flagged for review
    ├── parser_statistics.json  # Comprehensive coverage and quality statistics
    └── human_inspection_sample.csv # 120 curated representative samples
```

---

## 4. Pipeline Execution Workflow

```mermaid
graph TD
    Raw[Raw Messy Address] --> Translit[Script Detection & Transliteration]
    Translit --> Norm[Normalization & Tokenization]
    Norm --> Pin[Pincode Extraction]
    Norm --> Town[Town Resolution]
    Norm --> Loc[Gazetteer Locality Resolution]
    Norm --> Ward[Ward Extraction]
    Norm --> Street[Street Extraction]
    Norm --> HNo[House Number Extraction]
    Pin & Town & Loc & Ward & Street & HNo --> Protect[Precedence & Conflict Resolution]
    Protect --> Landmark[Landmark Extraction on Remaining Text]
    Landmark --> Other[Other Details Capture]
    Other --> Weak[Weak Spans & BIO Token Alignment]
    Weak --> Filter[Quality & Confidence Filtering]
    Filter --> TrainData[High-Confidence Training Subset]
    TrainData --> MuRIL[MuRIL Transformer NER Fine-Tuning]
    MuRIL --> Struct[Structured Output Entities]
```

---

## 5. How to Run Everything

### Step 1: Run Automated Unit Tests
```bash
python -m pytest tests/ -v
```

### Step 2: Run the Weak-Supervision Parser Pipeline
Generates parsed CSVs, weak label datasets, BIO training data, and statistical reports:
```bash
python -m src.main
```

### Step 3: Train the MuRIL NER Model
Trains the transformer on the high-confidence weak supervision dataset with group-stratified splits:
```bash
python -m src.train_ner
```

### Step 4: Run Inference on Any Address
Run the inference engine from CLI or import in Python:
```bash
python -m src.inference "6th Cross, 5th Main, ಚರ್ಚ್ ಹತ್ತಿರ, Kuvempu Layt, Kaveripura - 960102"
```

In Python:
```python
from src.inference import AddressNERPredictor

predictor = AddressNERPredictor()
result = predictor.predict("217/2, Gali no-4, opp Water Tank, Ambedkar Nagar, Devgarh Nagar - 970204")
print(result["structured_address"])
# Output:
# {
#   'house_number': '217/2',
#   'streets': ['Gali no-4'],
#   'ward': None,
#   'landmarks': ['opp Water Tank'],
#   'locality': 'Ambedkar Nagar',
#   'town': 'Devgarh Nagar',
#   'pincode': '970204'
# }
```

---

## 6. Systematic Improvements Over Baseline

| Metric | Baseline Prototype | Our Pipeline | Improvement |
| :--- | :--- | :--- | :--- |
| **Total T1/T2/T3 Addresses** | 2,880 | 2,880 | Benchmark subset |
| **Valid Pincodes** | 2,670 | 2,670 | 100% precision isolated 6-digit match |
| **Resolved Localities** | 2,415 | 2,598 | **+183 (+7.6%)** |
| **Unresolved Localities** | 465 | 282 | **-39.4% reduction in unparsed** |
| **House Numbers Extracted** | 1,173 | 1,962 | **+789 (+67.3%)** |
| **Streets Extracted** | 1,694 (Street/Ward) | 2,398 | **+41.6% coverage** |
| **Wards Extracted** | Included above | 240 | Dedicated extraction |
| **Landmarks Extracted** | 1,658 | 2,016 | **+358 (+21.6%)** |
| **Total Labeled Spans** | - | 15,416 | Multi-span coverage |
| **High-Confidence Tokens** | - | 49,363 | Silver training set |
