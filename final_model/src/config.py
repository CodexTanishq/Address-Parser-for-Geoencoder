from pathlib import Path

# Base Paths strictly within final_model/
FINAL_MODEL_DIR = Path(__file__).resolve().parent.parent
DATA_REF_DIR = FINAL_MODEL_DIR / "data_ref"
SRC_DIR = FINAL_MODEL_DIR / "src"
MODELS_DIR = FINAL_MODEL_DIR / "models"
OUTPUTS_DIR = FINAL_MODEL_DIR / "outputs"

# Reference Data Files
ADDRESSES_CSV = DATA_REF_DIR / "addresses.csv"
TOWNS_CSV = DATA_REF_DIR / "towns.csv"
LOCALITIES_CSV = DATA_REF_DIR / "localities.csv"
LANDMARKS_POI_CSV = DATA_REF_DIR / "landmarks_poi.csv"
SPLITS_CSV = DATA_REF_DIR / "splits.csv"
FIELD_VISITS_CSV = DATA_REF_DIR / "field_visits.csv"
BASELINE_GEOCODES_CSV = DATA_REF_DIR / "baseline_geocodes.csv"
SURVEYED_ADDRESSES_CSV = DATA_REF_DIR / "surveyed_addresses.csv"
MASTER_DATA_V5_CSV = DATA_REF_DIR / "master_data_3_locations_v5.csv"

# Model Checkpoints & Output Files
MURIL_NER_MODEL_DIR = MODELS_DIR / "muril_address_ner_final"
TFIDF_TOWN_MODEL_PATH = MODELS_DIR / "tfidf_town_clf.joblib"
TFIDF_LOCALITY_MODEL_PATH = MODELS_DIR / "tfidf_locality_clf.joblib"

PREDICTIONS_TEST_CSV = OUTPUTS_DIR / "predictions_test.csv"
EVALUATION_METRICS_JSON = OUTPUTS_DIR / "evaluation_metrics.json"

# Fixed random seed
RANDOM_SEED = 42

# NER Tags (Compact Tag Set)
# TOWN, LOCALITY, LANDMARK, RELATION
NER_ENTITIES = ["TOWN", "LOCALITY", "LANDMARK", "RELATION"]
BIO_LABELS = ["O"]
for ent in NER_ENTITIES:
    BIO_LABELS.extend([f"B-{ent}", f"I-{ent}"])

LABEL2ID = {label: i for i, label in enumerate(BIO_LABELS)}
ID2LABEL = {i: label for i, label in enumerate(BIO_LABELS)}

# Hyperparameters
TRANSFORMER_MODEL_NAME = "google/muril-base-cased"
MAX_SEQ_LENGTH = 128
TRAIN_BATCH_SIZE = 16
EVAL_BATCH_SIZE = 32
LEARNING_RATE = 3e-5
NUM_EPOCHS = 8
