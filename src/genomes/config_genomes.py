from src import config_main as main_cfg
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg

GENOMES_DIR = main_cfg.P_TRIAL3 / "genomes"
MANIFEST_PATH = GENOMES_DIR / "manifest.tsv"
STAGING_DIR = GENOMES_DIR / ".staging"

UNIT_MEASURES_PATH = vmatch_cfg.STEP6_UNIT_MEASURES_PATH
CONTEXTS_PATH = vmatch_cfg.STEP2_CONTEXTS_PATH

DATASETS_INCLUDE = ("genome", "gff3", "protein", "seq-report")
BATCH_SIZE = 200

FILES = {"fna": ("genomic.fna", "*_genomic.fna"), "gff": ("genomic.gff", "genomic.gff"),
         "faa": ("protein.faa", "protein.faa"), "sequence_report": ("sequence_report.jsonl", "sequence_report.jsonl")}
REPORT_NAME = "assembly_report.json"
