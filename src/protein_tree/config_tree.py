from dataclasses import dataclass
from pathlib import Path

from src import config_main as main_cfg
from src.stage02_domains import config_domains as domains_cfg

OUT_DIR = main_cfg.P_TRIAL3 / "protein_tree"

TREES = ("core", "full")
HANDOFF_DIR = OUT_DIR / "handoff"
RETURNED_DIR = OUT_DIR / "returned"
TREES_DIR = OUT_DIR / "trees"

FULL_TREE_VERSION = 3
TREE_NAMES = {"core": "core", "full": f"full_protein_tree_v{FULL_TREE_VERSION}"}
TREE_PATHS = {tree: TREES_DIR / f"{name}.treefile" for tree, name in TREE_NAMES.items()}

FROZEN_TREES = ("core",)

SAMPLES_DIR = OUT_DIR / "samples"
SAMPLES_PATH = SAMPLES_DIR / "samples.tsv"
PRUNED_TREE_PATHS = {name: SAMPLES_DIR / f"{name}.pruned.treefile" for name in TREES}

REPRESENTATIVE_FASTA_PATH = domains_cfg.DPAM_INPUT_FASTA_PATH
ANCHOR_FASTA_PATH = domains_cfg.DPAM_ANCHOR_INPUT_FASTA_PATH
DPAM_DOMAINS_TABLE_PATH = domains_cfg.DPAM_DOMAINS_TABLE_PATH
PROTEIN_ARCHITECTURES_PATH = domains_cfg.DPAM_PROTEIN_ARCHITECTURES_PATH

PROTEIN_STATUS_PATH = OUT_DIR / "intact" / "protein_status.tsv"

CORE_TOKENS = ("RecG", "Schlafen")

CORE_COMPLETE_SHARE = 0.85
CORE_TRUNCATED_SHARE = 0.70

PILOT_DIR = Path(main_cfg.p_trial1) / "DATA" / "stage3_files_msa" / "iter_0"
PILOT_TREE_PATH = PILOT_DIR / "tree" / "iter_0_d_msa.fasta.treefile"
PILOT_FASTA_PATH = PILOT_DIR / "source" / "iter_0_c_combined.fasta"
PILOT_PROTEINS_PATH = PILOT_DIR / "source" / "iter_0_b_mapped3.csv"

RELATIVE_ANCHORS = ("LacN", "CfeN", "MfoN")
RELATIVES_PER_ANCHOR = 12
RELATIVE_MIN_LENGTH_SHARE = 0.85
RELATIVES_PATH = OUT_DIR / "relatives" / "relatives.tsv"

@dataclass(frozen=True)
class CorePartition:

    key: str
    token: str
    description: str

CORE_PARTITIONS = (CorePartition("schlafen", "Schlafen", "Schlafen domain, one per protein where called."),
                   CorePartition("recg", "RecG", "RecG domain, one per protein where called."),)
