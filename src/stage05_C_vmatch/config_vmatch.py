import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src import config_main as main_cfg
from src.protein_tree import config_tree as tree_cfg
from src.stage02_domains import config_domains as domains_cfg
from src.stage03_concordance import config_concordance as concordance_cfg

load_dotenv()

OUT_DIR = main_cfg.P_TRIAL3 / "stage05_C_vmatch"
SELECTIONS_DIR = OUT_DIR / "selections"
DOWNLOADS_DIR = OUT_DIR / "downloads"
IPG_SUPPLEMENT_DIR = DOWNLOADS_DIR / "ipg_supplement"
SEGMENT_DOWNLOAD_DIR = DOWNLOADS_DIR / "segments"
EXTRACTIONS_DIR = OUT_DIR / "extractions"
PROTEINS_DIR = OUT_DIR / "proteins"
SUMMARIES_DIR = OUT_DIR / "summaries"
LOGS_DIR = OUT_DIR / "logs"

DPAM_PROTEIN_ARCHITECTURES_PATH = domains_cfg.DPAM_PROTEIN_ARCHITECTURES_PATH
HTH_CLASSES = domains_cfg.DPAM_HTH_CLASSES
CONCORDANCE_PATH = concordance_cfg.CONCORDANCE_PATH
BLAST_ALIGNMENT_PATH = main_cfg.P_TRIAL3 / "X4Y60Y89016-Alignment.csv"
FASTA_PATH = Path(main_cfg.path_combined_addendum)
MAPPED_PATH = Path(main_cfg.path_mapped_1)
ADDENDUM_PATH = Path(main_cfg.path_addendum_csv)
IPG_DIR = Path(main_cfg.path_IPG)
ASSEMBLY_SUMMARY_PATHS = [
    Path(main_cfg.path_bacteria_assembly_summary),
    Path(main_cfg.path_archaea_assembly_summary),
]

MIN_CONTEXTS = 1
MAX_CONTEXTS = 100
SELECTION_RANDOM_SEED = 42

CANONICAL_HTH_ARCHITECTURE = "Schlafen -- RecG -- HTH > GLPR"
CANONICAL_TWO_HTH_ARCHITECTURE = "Schlafen -- RecG -- HTH -- HTH > GLPR"
CANONICAL_WITHOUT_HTH_ARCHITECTURE = "Schlafen -- RecG > GLPR"
SCHLAFEN_DOMAIN = "Schlafen"

CONTEXT_AVAILABILITY_PATH = SELECTIONS_DIR / "context_availability.tsv"
CONTEXT_RECORDS_PATH = SELECTIONS_DIR / "context_records.tsv"
CANDIDATE_SUMMARY_PATH = SELECTIONS_DIR / "candidate_summary.tsv"
FULL_MANIFEST_PATH = SELECTIONS_DIR / "candidate_manifest.full.jsonl"
ACTIVE_MANIFEST_PATH = SELECTIONS_DIR / "candidate_manifest.active.jsonl"

FLANK_LENGTH = 1000
MIN_FLANK_LENGTH = 100
TRANSLATION_TABLES = (11, 4, 25)

FLANK_CLUSTER_METHOD = "edlib_shw_prefix_star"
FLANK_CLUSTER_MIN_PREFIX_SIMILARITY = 0.98
FLANK_CLUSTER_AMBIGUITY_POLICY = "non_acgt_counts_as_edit"

STEP2_CONTEXTS_PATH = SUMMARIES_DIR / "step2_contexts.tsv"
STEP2_PROTEINS_PATH = SUMMARIES_DIR / "step2_proteins.tsv"
STEP2_LEFT_CLUSTERS_PATH = SUMMARIES_DIR / "step2_left_clusters.tsv"
STEP2_RIGHT_CLUSTERS_PATH = SUMMARIES_DIR / "step2_right_clusters.tsv"

MKVTREE_BIN = Path(shutil.which("mkvtree") or "mkvtree")
VMATCH_BIN = Path(shutil.which("vmatch") or "vmatch")
VMATCHSELECT_BIN = Path(shutil.which("vmatchselect") or "vmatchselect")

VMATCH_PREFIX_LENGTH = 2
VMATCH_MIN_LENGTH = 10
VMATCH_MAX_SPACER = 300

VMATCH_WITHIN_SEED_LENGTH = 10
VMATCH_WITHIN_HXDROP = 1
VMATCH_WITHIN_MIN_IDENTITY = 80
VMATCH_WITHIN_MAX_EVALUE = 0.1
VMATCHSELECT_WITHIN_MAX_EVALUE = 1e-2

VMATCH_MAX_EDIT_DISTANCE = 4
VMATCH_MIN_IDENTITY = 85
VMATCH_MAX_EVALUE = 0.1

VMATCHSELECT_MIN_LENGTH = 10
VMATCHSELECT_MIN_IDENTITY = 85
VMATCHSELECT_MAX_EVALUE = 1e-2

TRF_BIN = Path(shutil.which("trf409.linux64") or "trf409.linux64")
TRF_MATCH_WEIGHT = 2
TRF_MISMATCH_PENALTY = 7
TRF_INDEL_PENALTY = 7
TRF_MATCH_PROBABILITY = 80
TRF_INDEL_PROBABILITY = 10
TRF_MIN_SCORE = 50
TRF_MAX_PERIOD = 500
TRF_MIN_PERIOD = 10
TRF_MIN_FULL_COPIES = 2
TRF_NORMALIZATION_TAU_SPAN = 0.20
TRF_NORMALIZATION_TAU_SCORE = 0.30
TRF_NORMALIZATION_MIN_RECIPROCAL_OVERLAP = 0.70
TRF_VMATCH_MIN_ARM_COVERAGE = 0.25
TRF_VMATCH_MIN_OVERLAP_BP = 2

TRF_NORMALIZATION_POLICY = "span_score_smallest_period_v1"
TRF_VMATCH_SUPPRESSION_POLICY = "legacy_array_span_both_arms_v1"

V5_POLICY = "architecture_inference_deterministic_v5"
V5_SETTINGS = dict(min_copies=2, min_unit_length=10, min_total_core_bases=20, refinement="trim-and-extend",
                   selection="score-aware", objective="column", target_identity=None)

STEP3_CALIBRATION_PATH = SUMMARIES_DIR / "step3_vmatch_calibration.tsv"
STEP3_CONTEXTS_PATH = SUMMARIES_DIR / "step3_vmatch_contexts.tsv"
STEP3_MATCHES_PATH = SUMMARIES_DIR / "step3_vmatch_matches.tsv"
STEP3_RUNS_PATH = SUMMARIES_DIR / "step3_vmatch_runs.tsv"
STEP3_TRF_ARRAYS_PATH = SUMMARIES_DIR / "step3_trf_arrays.tsv"
STEP3_TRF_RUNS_PATH = SUMMARIES_DIR / "step3_trf_runs.tsv"
STEP3_TRF_NORMALIZATION_FINGERPRINT_PATH = (
    SUMMARIES_DIR / "step3_trf_normalization.json"
)

STEP4_ARCHITECTURES_PATH = SUMMARIES_DIR / "step4c_architectures.tsv"
STEP4_ARCHITECTURE_AUDIT_PATH = SUMMARIES_DIR / "step4c_architecture_audit.jsonl"
STEP4_VMATCH_SUPPRESSION_LEDGER_PATH = (
    SUMMARIES_DIR / "step4c_vmatch_suppression.jsonl"
)
STEP4_EVIDENCE_PATH = SUMMARIES_DIR / "step4c_evidence.tsv"
STEP4_UNIT_COMPOSITION_PATH = SUMMARIES_DIR / "step4c_unit_composition.tsv"
STEP4_MARKOV_PSEUDOCOUNT = 0.5
STEP4_LOG_PATH = LOGS_DIR / "step4c_architectures.log"

STEP5_CANDIDATES_PATH = CONTEXT_AVAILABILITY_PATH
STEP5_COPIES_PATH = SUMMARIES_DIR / "step5c_copies.tsv"
STEP5_ARCHITECTURES_PATH = SUMMARIES_DIR / "step5c_architectures.tsv"
STEP5_COPY_MATCHES_PATH = SUMMARIES_DIR / "step5c_copy_matches.tsv"
STEP5_ARCHITECTURE_RECURRENCE_PATH = (
    SUMMARIES_DIR / "step5c_architecture_recurrence.tsv"
)
STEP5_PROTEIN_SUMMARY_PATH = SUMMARIES_DIR / "step5c_protein_summary.tsv"
STEP5_MEASURED_ARCHITECTURES_PATH = SUMMARIES_DIR / "step5c_measured_architectures.tsv"
STEP5_MEASURED_EVIDENCE_PATH = SUMMARIES_DIR / "step5c_measured_evidence.tsv"
STEP5_EVIDENCE_RECURRENCE_PATH = SUMMARIES_DIR / "step5c_evidence_recurrences.tsv"
STEP5_EVIDENCE_THRESHOLD_SWEEP_PATH = SUMMARIES_DIR / "step5c_evidence_threshold_sweep.tsv"
STEP5_TANDEM_EVALUE_BOUND_PATH = SUMMARIES_DIR / "step5c_tandem_evalue_bound.tsv"
STEP5_TRF_ADMISSIBLE_EVALUE_PATH = SUMMARIES_DIR / "step5c_trf_admissible_evalue.tsv"
STEP5_TWO_COPY_ARRAY_EVALUE_PATH = SUMMARIES_DIR / "step5c_two_copy_array_evalue.tsv"
STEP5_TWO_COPY_ARRAY_SENSITIVITY_PATH = SUMMARIES_DIR / "step5c_two_copy_array_sensitivity.tsv"
STEP5_LOG_PATH = LOGS_DIR / "step5c_analysis.log"

STEP5_SIMILARITY_THRESHOLD = 0.95
STEP5_MIN_INDEPENDENT_MATCHES = 2
STEP5_MIN_RECURRENCE_UNITS = 5
STEP5_NEAR_CDS_CUTOFF = 200

STEP5_COPY_LENGTH_BINS = (
    (1, 9, "1-9"),
    (10, 15, "10-15"),
    (16, 30, "16-30"),
    (31, 60, "31-60"),
    (61, None, "61+"),
)
STEP5_EDGE_SEPARATION_BINS = (
    (1, 20, "1-20"),
    (21, 100, "21-100"),
    (101, 300, "101-300"),
    (301, None, "301+"),
)
STEP5_DISTANCE_BINS = (
    (1, 50, "1-50"),
    (51, 200, "51-200"),
    (201, 500, "201-500"),
    (501, None, "501+"),
)

@dataclass(frozen=True)
class RepeatSet:

    name: str
    kinds: tuple[str, ...]
    max_pair_evalue: float
    sources: tuple[str, ...] = ("arch", "evidence")

STEP5_REPEAT_KINDS = ("tandem_array", "direct", "palindromic_inverted")
STEP5_MAX_PAIR_EVALUE = 1e-3
STEP5_STRICT_MAX_PAIR_EVALUE = 1e-5
STEP5_REPEAT_SETS = (RepeatSet("all", STEP5_REPEAT_KINDS, STEP5_MAX_PAIR_EVALUE),
                     RepeatSet("tandem", ("tandem_array",), STEP5_MAX_PAIR_EVALUE),
                     RepeatSet("direct", ("direct",), STEP5_MAX_PAIR_EVALUE),
                     RepeatSet("inverted", ("palindromic_inverted",), STEP5_MAX_PAIR_EVALUE, sources=("evidence",)),
                     RepeatSet("all_strict", STEP5_REPEAT_KINDS, STEP5_STRICT_MAX_PAIR_EVALUE, sources=("evidence",)))
STEP5_PRIMARY_REPEAT_SET = "all"

if STEP5_PRIMARY_REPEAT_SET not in {repeat_set.name for repeat_set in STEP5_REPEAT_SETS}:

    raise ValueError(f"STEP5_PRIMARY_REPEAT_SET {STEP5_PRIMARY_REPEAT_SET!r} names no declared repeat set")

STEP5_MIN_COPY_LENGTH = 10
STEP5_PROXIMAL_CUTOFF = 1000
STEP5_UNIT_WINDOWS = (STEP5_NEAR_CDS_CUTOFF, 500, STEP5_PROXIMAL_CUTOFF)
STEP5_MIN_UNITS = 1

STEP5_HTH_PRESENCE = {"no_HTH": "no_HTH", "HTH+": "HTH_present", "HTH++": "HTH_present"}

STEP5_RESULT_UNITS = ("protein", "protein_cluster_weighted", "protein_cluster_representative")

STEP5_RESULT_VIEWS = ("hth_presence", "hth_class", "hth_families", "architecture",
                      "architecture_source")

RESULT_MEASURE_PATHS = {view: SUMMARIES_DIR / f"result_measures_by_{view}.tsv"
                          for view in STEP5_RESULT_VIEWS}

STEP6_SAMPLES_PATH = tree_cfg.SAMPLES_PATH
STEP6_SAMPLE_SUMMARY_PATH = SUMMARIES_DIR / "step6c_sample_summary.tsv"
STEP6_SAMPLE_SUMMARY_MANIFEST_PATH = SUMMARIES_DIR / "step6c_sample_summary.json"
STEP6_EVIDENCE_RECURRENCE_PATH = SUMMARIES_DIR / "step6c_evidence_recurrences.tsv"
STEP6_NEAR_COPIES_PATH = SUMMARIES_DIR / "step6c_near_copies.tsv"
STEP6_UNIT_MEASURES_PATH = SUMMARIES_DIR / "step6c_unit_measures.tsv"
STEP6_LOG_PATH = LOGS_DIR / "step6c_summary.log"
STEP6_POOLED_PROTEINS_FIELD = "pooled_proteins"
STEP6_NEAR_COPIES_FIELD = "near_copies_removed"

BACKGROUND_DIR = OUT_DIR / "background"
BACKGROUND_PANELS = {"pilot": {"genes_per_genome": 10, "assemblies": "GCF_"}, "full": {"genes_per_genome": 20, "assemblies": "GCF_"},
                     "genbank": {"genes_per_genome": 20, "assemblies": "GCA_"}}
BACKGROUND_PILOT_GENOMES = 30
BACKGROUND_SEED = 20261009

NCBI_EMAIL = os.environ.get("NCBI_EMAIL")
NCBI_API_KEY = os.environ.get("NCBI_API_KEY")
