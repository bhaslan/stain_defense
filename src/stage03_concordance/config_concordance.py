from src import config_main as main_cfg

FASTA_PATH = main_cfg.path_esmc_fasta
MODEL_ID = "esmc-6b-2024-12"
EMBEDDING_LAYER = "safe"
MEAN_POOL_TOKEN_SCOPE = "all_tokens"

FORGE_URL = "https://biohub.ai"
TOKEN_ENV_VAR = "ESM_API_KEY"
API_PARALLEL_REQUESTS = 10
MAX_RECORDS = None

EMBEDDING_OUT_DIR = main_cfg.P_STAGE03 / "esmc_embeddings"
_ESMC_API_NUM_LAYERS = {
    "esmc-300m-2024-12": 30,
    "esmc-600m-2024-12": 36,
    "esmc-6b-2024-12": 80,
}

def _resolve_layer_index(model_id, layer):
    n_layers = _ESMC_API_NUM_LAYERS.get(model_id)
    if layer == "safe":
        if n_layers is None:
            raise ValueError(f"Unknown ESMC model for layer='safe': {model_id}")
        return n_layers - 1
    if layer == "last":
        if n_layers is None:
            raise ValueError(f"Unknown ESMC model for layer='last': {model_id}")
        return n_layers
    if isinstance(layer, int):
        if layer < 0:
            if n_layers is None:
                raise ValueError(f"Unknown ESMC model for negative layer: {model_id}")
            return n_layers + 1 + layer
        return layer
    raise ValueError("EMBEDDING_LAYER must be an integer, 'safe', or 'last'.")

EMBEDDING_LAYER_INDEX = _resolve_layer_index(MODEL_ID, EMBEDDING_LAYER)
MODEL_FILE_STEM = MODEL_ID.replace("-", "_")
POOL_FILE_STEM = f"{MODEL_FILE_STEM}_layer{EMBEDDING_LAYER_INDEX}_meanpooled_{MEAN_POOL_TOKEN_SCOPE}"
PER_POSITION_ARCHIVE_PATH = EMBEDDING_OUT_DIR / f"combined_addendum.{MODEL_FILE_STEM}_layer{EMBEDDING_LAYER_INDEX}_positions_raw.h5"
MEANPOOLED_RAW_PATH = EMBEDDING_OUT_DIR / f"combined_addendum.{POOL_FILE_STEM}_raw.h5"
MEANPOOLED_L2_PATH = EMBEDDING_OUT_DIR / f"combined_addendum.{POOL_FILE_STEM}_l2.h5"
EMBEDDING_ARCHIVE_PATH = MEANPOOLED_L2_PATH

EMBEDDING_SPACE = "raw"

NEIGHBOR_K = 20
HTH_RICH = 0.70
HTH_POOR = 0.30
GLPR_SIG_KEY = "PANTHER:PTHR30595"

LENGTH_SENSITIVITY_K_VALUES = [10, NEIGHBOR_K, 50]
LENGTH_SENSITIVITY_BINS = 4
LENGTH_SENSITIVITY_PERMUTATIONS = 1000
LENGTH_SENSITIVITY_RANDOM_STATE = 0

OUT_DIR = main_cfg.P_STAGE03 / "esmc_concordance"
ANALYSIS_MODEL_NAME = MODEL_ID.replace("-", "_")
ANALYSIS_NAME = f"{ANALYSIS_MODEL_NAME}_{EMBEDDING_SPACE}_knn{NEIGHBOR_K}"

CONCORDANCE_PATH = OUT_DIR / f"concordance_{ANALYSIS_NAME}.tsv"
KNN_EDGES_PATH = OUT_DIR / f"knn_edges_{ANALYSIS_NAME}.tsv"
LENGTH_SENSITIVITY_PATH = OUT_DIR / f"concordance_length_sensitivity_{ANALYSIS_NAME}.tsv"
