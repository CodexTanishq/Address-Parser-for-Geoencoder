from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = PROJECT_ROOT / "models" / "hierarchical_address_embedding"

# Data Files
ADDRESSES_CSV = DATA_DIR / "addresses.csv"
TOWNS_CSV = DATA_DIR / "towns.csv"
LOCALITIES_CSV = DATA_DIR / "localities.csv"
PARSED_ADDRESSES_CSV = OUTPUTS_DIR / "parsed_addresses.csv"

# Pretrained Backbone
PRETRAINED_MODEL_NAME = "google/muril-base-cased"

# Architecture Hyperparameters
MAX_SEQ_LENGTH = 128
PROJECTION_DIM = 256  # Common shared embedding space
POOLING_MODE = "attention"  # "cls", "mean", "attention"

# Loss Weights (configurable λ parameters)
LAMBDA_TOWN = 1.0
LAMBDA_LOCALITY = 1.0
LAMBDA_LANDMARK = 0.5
LAMBDA_HIERARCHY = 0.3
LAMBDA_ALIGNMENT = 0.2

# Contrastive Loss Hyperparameters
TEMPERATURE = 0.07
TRIPLET_MARGIN = 0.3

# Training Settings
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 3
WEIGHT_DECAY = 0.01
RANDOM_SEED = 42

# Device Configuration
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Compatibility Aliases
MODEL_NAME = PRETRAINED_MODEL_NAME
POOLING_TYPE = POOLING_MODE
MODEL_SAVE_DIR = MODELS_DIR
PREPROCESSING_MODE = "normalized_transliterated"
USE_PINCODE_FEATURE = False

# Confidence & Ambiguity Thresholds
SIMILARITY_THRESHOLD = 0.50
MARGIN_THRESHOLD = 0.05
