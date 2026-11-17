from pathlib import Path

from src import config_main as main_cfg
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg

OUT_DIR = main_cfg.P_TRIAL3 / "stage06_repeat_enrichment"
KINSHIP_DIR = OUT_DIR / "kinship"

KINSHIP_TREES = ("full",)
BRANCH_KINSHIP_PATH = KINSHIP_DIR / "core.tsv"

UNIT_MEASURES_PATH = vmatch_cfg.STEP6_UNIT_MEASURES_PATH

HTH_CLASSES = ("no_HTH", "HTH+", "HTH++")

BAYES_ENV = "bayes_env"
GLMM_ENV = "glmm_env"
MODELS_DIR = OUT_DIR / "models"
BAYES_BUILD_DIR = MODELS_DIR / "build"
BAYES_SETTINGS = {"chains": 4, "warmup": 1000, "sampling": 1000, "seed": 20261008}

COMPARISON_DIR = OUT_DIR / "comparison"

BACKGROUND_PANEL = "full"
BACKGROUND_COMPARISON_PATH = COMPARISON_DIR / "comparison.tsv"
BACKGROUND_FLANKS_PATH = COMPARISON_DIR / "focal_flanks.tsv"
BACKGROUND_WINDOWS = (200, 500)
BACKGROUND_BOOTSTRAP = 2000
BACKGROUND_GC_BINS = 10
BACKGROUND_BRANCHES = 8
BACKGROUND_BRANCHES_PATH = COMPARISON_DIR / "by_branch.tsv"
BACKGROUND_SEED = vmatch_cfg.BACKGROUND_SEED
BACKGROUND_BAYES_SCRIPT = Path(__file__).with_name("bayes_background.R")
BACKGROUND_GLMM_SCRIPT = Path(__file__).with_name("glmm_background.R")

BACKGROUND_MODEL_PANELS = ("full", "genbank")
BACKGROUND_MODEL_FITS = (("all", 200), ("all", 500), ("tandem", 200), ("tandem", 500), ("direct", 200), ("inverted", 200))
BACKGROUND_MODEL_TABLE_PATH = COMPARISON_DIR / "model_table.tsv"

SIDE_RINGS = tuple(zip((0, *vmatch_cfg.STEP5_UNIT_WINDOWS[:-1]), vmatch_cfg.STEP5_UNIT_WINDOWS))
BACKGROUND_SIDES_PATH = COMPARISON_DIR / "by_side.tsv"
POOLED_PANEL = "pooled"
POOLED_PANELS = ("full", "genbank")

UNIT_SIZE_WINDOW = 200
UNIT_SIZE_CLASSES = ((10, 12), (13, 39), (40, vmatch_cfg.TRF_MAX_PERIOD))
UNIT_SPECTRUM_LARGEST = 150
UNIT_SIZE_PATH = COMPARISON_DIR / "by_unit_size.tsv"
UNIT_SPECTRUM_PATH = COMPARISON_DIR / "unit_size_spectrum.tsv"

GC_GAP_EDGES = (50, 100, 200, 400)
FLANK_GC_PATH = COMPARISON_DIR / "flank_gc.tsv"
FLANK_GC_GAPS_PATH = COMPARISON_DIR / "flank_gc_by_gap.tsv"

LINEAGE_WINDOW = 200
LINEAGE_MIN_FLANKS = 4
LINEAGE_PERMUTATIONS = 2000
ALL_OR_NONE_PATH = COMPARISON_DIR / "all_or_none.tsv"
FOLLOWS_PATH = COMPARISON_DIR / "follows_the_protein.tsv"

TERMINATOR_REPEAT_SET = "inverted"
TERMINATOR_WINDOW = 200
OPERON_GAP = 50
TERMINATOR_CHECK_PATH = COMPARISON_DIR / "terminator_check.tsv"

FAMILY_MIN_SAMPLES = 20
FAMILY_MIN_GENERA = 10
FAMILY_COMPARISON_PATH = COMPARISON_DIR / "by_family.tsv"
FAMILY_MODELS_DIR = MODELS_DIR / "families"
FAMILY_MODEL_FITS = (("tandem", 200), ("direct", 200), ("inverted", 200))
FAMILY_PARALLEL_FITS = 12
FAMILY_BAYES_SETTINGS = dict(BAYES_SETTINGS, adapt_delta=0.99)
FAMILY_MODEL_TABLE_PATH = COMPARISON_DIR / "family_model_table.tsv"
