from src import config_main as main_cfg

OUT_DIR = main_cfg.P_TRIAL3 / "stage07_figures"
WORKFLOW_DIR = OUT_DIR / "fig-5A_workflow"
TREE_DIR = OUT_DIR / "fig-5B_tree"
TREE_BUILD_DIR = TREE_DIR / "build"
ESMC_DISTANCE_DIR = OUT_DIR / "fig-6G_esmc_distance"
ENRICHMENT_DIR = OUT_DIR / "fig-5E_enrichment"
FAMILY_ENRICHMENT_DIR = OUT_DIR / "fig-5F_family_enrichment"
REPEAT_SIZE_DIR = OUT_DIR / "fig-5D_repeat_size"
SIDE_PROFILE_DIR = OUT_DIR / "fig-5G_side_profile"
EVIDENCE_SWEEP_DIR = OUT_DIR / "evidence_sweep"
REPEAT_RECURRENCE_DIR = OUT_DIR / "repeat_recurrence"
CONCORDANCE_DIR = OUT_DIR / "concordance"
SAE_DIR = OUT_DIR / "sae"
HTH_COUNT_DIR = OUT_DIR / "fig-5C_hth_count"
HOMOLOG_TABLE_DIR = OUT_DIR / "table_S4_homologs"
FLANK_TABLE_DIR = OUT_DIR / "table_S5_flanks"
DPI = 300

HTH_CLASS_COLOURS = {"no_HTH": "#D9D9D9", "HTH+": "#6BADD6", "HTH++": "#084594"}

PHYLUM_COLOURS = {"Bacillota": "#E69F00", "Pseudomonadota": "#56B4E9", "Bacteroidota": "#009E73", "Actinomycetota": "#CC79A7",
                  "Cyanobacteriota": "#F0E442", "Methanobacteriota": "#D55E00", "Fusobacteriota": "#0072B2", "other": "#999999"}

CLADE_COLOURS = ("#E41A1C", "#377EB8", "#4DAF4A", "#984EA3", "#FF7F00", "#A65628", "#F781BF", "#17BECF")

ENRICHMENT_WINDOW = 200
ENRICHMENT_REPEAT_SETS = ("tandem", "direct", "inverted")
ENRICHMENT_CLASSES = ("HTH+", "HTH++")
ENRICHMENT_SIZE = (2.45, 3.5)
TERMINATOR_SIZE = (3.5, 3.3)
ENRICHMENT_PANELS = {"enrichment_panel.png": ("full",), "MAIN_enrichment_panel_refseq_and_genbank.png": ("full", "genbank")}
TERMINATOR_PANELS = {"full": "terminator_panel.png", "genbank": "terminator_panel_genbank_supplement.png"}

FAMILY_COLUMN_REPEAT_SETS = ("tandem", "direct")
STAN_HTH_FAMILY = "FUR"
FAMILY_COLUMN_SIZE = (1.77, 5.31)
FAMILY_COLUMN_DPI = 600
FAMILY_SUPPLEMENT_SIZE = (7.0, 4.6)
FAMILY_ENRICHMENT_PATH = FAMILY_ENRICHMENT_DIR / "MAIN_family_enrichment_panel.png"
FAMILY_SUPPLEMENT_PATH = FAMILY_ENRICHMENT_DIR / "family_enrichment_supplement.png"

SIDE_PROFILE_SIZE = (7.0, 3.9)
SIDE_PROFILE_PANELS = {"full": "side_profile.png", "genbank": "side_profile_genbank_supplement.png"}
SIDE_PROFILE_LABEL_P = 0.05
SIDE_HEAT_WIDTH = 3.6
SIDE_HEAT_PANELS = {"full": "MAIN_side_heat_tracks.png", "genbank": "side_heat_tracks_genbank_supplement.png", "pooled": "side_heat_tracks_pooled.png"}

REPEAT_CHARACTER_DIR = OUT_DIR / "fig-5HI_repeat_character"
REPEAT_CHARACTER_PANELS = {"units": ("MAIN_repeat_character_panel_a.png", (3.5, 3.5)), "lineages": ("repeat_character_panel_b.png", (7.0, 2.6)),
                           "gc": ("repeat_character_panel_c_flanks.png", (3.2, 2.9)), "context_gc": ("MAIN_repeat_character_panel_c_all.png", (0.75, 1.75))}
REPEAT_CHARACTER_BUILD_DIR = REPEAT_CHARACTER_DIR / "build"
REPEAT_CHARACTER_C_ALL_DPI = 1200
CONTEXT_GC_WORKERS = 16
REPEAT_CHARACTER_SIDES = {"HTH+": "left", "HTH++": "right"}
REPEAT_CHARACTER_MIRROR_FLOOR = 1.0

REPEAT_SIZE_SIZE = (7.0, 4.4)
REPEAT_SIZE_PAPER_WINDOW = 200
REPEAT_SIZE_PAPER_SIZE = (2.6, 3.9)
REPEAT_SIZE_PAPER_FIGURES = {"MAIN_repeat_size_paper.png": ("all", ("all", "tandem", "direct")),
                             "repeat_size_paper_tandem.png": ("tandem", ("tandem",)),
                             "repeat_size_paper_direct.png": ("direct", ("direct",))}

HTH_COUNT_SIZE = (4.0, 3.0)
