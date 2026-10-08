from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

# Data Files
ADDRESSES_CSV = DATA_DIR / "addresses.csv"
TOWNS_CSV = DATA_DIR / "towns.csv"
LOCALITIES_CSV = DATA_DIR / "localities.csv"

# Output Files
PARSED_ADDRESSES_CSV = OUTPUTS_DIR / "parsed_addresses.csv"
WEAK_LABELS_CSV = OUTPUTS_DIR / "weak_labels.csv"
WEAK_SPANS_CSV = OUTPUTS_DIR / "weak_spans.csv"
BIO_DATASET_CSV = OUTPUTS_DIR / "bio_dataset.csv"
UNRESOLVED_ADDRESSES_CSV = OUTPUTS_DIR / "unresolved_addresses.csv"
AMBIGUOUS_ADDRESSES_CSV = OUTPUTS_DIR / "ambiguous_addresses.csv"
PARSER_STATISTICS_JSON = OUTPUTS_DIR / "parser_statistics.json"
HUMAN_INSPECTION_CSV = OUTPUTS_DIR / "human_inspection_sample.csv"

# NER Entity Classes
ENTITIES = [
    "HOUSE_NUMBER",
    "STREET",
    "WARD",
    "LANDMARK",
    "LOCALITY",
    "TOWN",
    "PINCODE",
]

# BIO Labels
BIO_LABELS = ["O"]
for entity in ENTITIES:
    BIO_LABELS.extend([f"B-{entity}", f"I-{entity}"])

LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(BIO_LABELS)}

# Confidence Levels
CONF_HIGH = "HIGH"
CONF_MEDIUM = "MEDIUM"
CONF_LOW = "LOW"
CONF_AMBIGUOUS = "AMBIGUOUS"

# Precedence order for non-overlapping conflict resolution
# Known structured entities take precedence before broader landmark spans
PRECEDENCE_ORDER = [
    "PINCODE",
    "TOWN",
    "LOCALITY",
    "WARD",
    "HOUSE_NUMBER",
    "STREET",
    "LANDMARK",
]

# Common Locality Abbreviations and Spelling Variants
LOCALITY_ABBREV_MAP = {
    "layt": "layout",
    "layou": "layout",
    "lyt": "layout",
    "ngr": "nagar",
    "nagr": "nagar",
    "ngar": "nagar",
    "colny": "colony",
    "clny": "colony",
    "col": "colony",
    "bdvne": "badavane",
    "badavne": "badavane",
    "shanti": "shanthi",
    "silvur": "silver",
    "neru": "nehru",
    "anjeneya": "anjaneya",
    "ganndhi": "gandhi",
    "baste": "basti",
    "idnira": "indira",
    "lakevew": "lakeview",
    "mhlla": "mohalla",
    "grdns": "gardens",
    "mdws": "meadows",
}

# Street Abbreviations
STREET_ABBREV_MAP = {
    "crs": "cross",
    "cr": "cross",
    "x": "cross",
    "mn": "main",
    "rd": "road",
    "ln": "lane",
    "st": "street",
}

# Random seed
RANDOM_SEED = 42

# Data Splitting
SPLITS_CSV = DATA_DIR / "splits.csv"

# Augmentation Settings
AUGMENTATION_PROB = 0.8
SHUFFLE_PROB = 0.7
NOISE_PROB = 0.3

# Transformer Model Settings
TRANSFORMER_MODEL_NAME = "google/muril-base-cased"
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUM_EPOCHS = 4

# Field Visits Data
FIELD_VISITS_CSV = DATA_DIR / "field_visits.csv"

# Model Directories & Checkpoints
MODEL_V1_DIR = MODELS_DIR / "muril_address_ner_v1_backup"
MODEL_V2_DIR = MODELS_DIR / "muril_address_ner_v2_multilandmark"
DEFAULT_MODEL_DIR = MODELS_DIR / "muril_address_ner"

# Spatial Coordinate & POI Data
LANDMARKS_POI_CSV = DATA_DIR / "landmarks_poi.csv"

# Spatial Dual-Encoder Model & Index Directories
EMBEDDING_BACKBONE = "google/muril-base-cased"
EMBEDDING_BATCH_SIZE = 32
EMBEDDING_LEARNING_RATE = 2e-5
EMBEDDING_NUM_EPOCHS = 4
SPATIAL_EMBEDDING_MODEL_V1_DIR = MODELS_DIR / "spatial_embedding_model_v1"
COORDINATE_INDEX_PATH = SPATIAL_EMBEDDING_MODEL_V1_DIR / "coordinate_index.pt"
COORDINATE_METADATA_PKL = SPATIAL_EMBEDDING_MODEL_V1_DIR / "metadata.pkl"
EMBEDDING_RETRIEVAL_METRICS_JSON = OUTPUTS_DIR / "embedding_retrieval_metrics.json"



