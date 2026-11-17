from pathlib import Path

from src import config_main as main_cfg

INTERPROSCAN_IMAGE = "interpro/interproscan:5.77-108.0"
INTERPROSCAN_MEMORY = "256g"
INTERPROSCAN_CPUS = 32
INTERPROSCAN_DB_PATH = Path(main_cfg.path_interproscan_db)
INTERPROSCAN_WORK_DIR = Path(main_cfg.p_interproscan)
INTERPROSCAN_INPUT_FASTA = Path(main_cfg.path_combined_addendum)
INTERPROSCAN_INPUT_NAME = main_cfg.fname
INTERPROSCAN_OUTPUT_DIR = INTERPROSCAN_WORK_DIR / "output"
INTERPROSCAN_TEMP_DIR = INTERPROSCAN_WORK_DIR / "temp"
INTERPRO_TSV_PATH = Path(main_cfg.path_interpro_tsv)

EXCLUDE_ANALYSES = [
    "MobiDBLite",
    "Coils",
    "ProSitePatterns",
    "ProSiteProfiles",
    "SMART",
    "PRINTS",
]

EXCLUDE_SIGNATURES = [
    "G3DSA:1.20.1480.30",
    "G3DSA:1.20.5.340",
    "G3DSA:6.10.10.130",
    "G3DSA:1.20.5.170",
    "G3DSA:1.20.1270.70",
]

ALIAS = {
    "Pfam:PF04326": "Schlafen",
    "Gene3D:G3DSA:3.30.950.30": "Schlafen",
    "Pfam:PF13749": "RecG",
    "Gene3D:G3DSA:3.30.565.60": "RecG",
    "Gene3D:G3DSA:1.10.10.10": "HTH",
    "SUPERFAMILY:SSF46785": "HTH",
    "Pfam:PF21247": "HTH",
    "Pfam:PF09339": "HTH",
}

LABEL_PRIORITY = [
    "RecG",
    "Schlafen",
    "HTH",
]

GLPR_SIG_KEY = "PANTHER:PTHR30595"

OUT_DIR = main_cfg.P_STAGE02 / "domain_analyzer"
COMBINED_FASTA_PATH = Path(main_cfg.path_combined_addendum)
ARCHITECTURE_FASTA_DIR = main_cfg.P_STAGE02 / "architectures_fastas"

DF_INTERPRO_PATH = OUT_DIR / "df_interpro.pkl"
DF_ARCHITECTURE_PATH = OUT_DIR / "df_architecture.pkl"
CANDIDATE_DOMAIN_EQUIVALENCES_PATH = OUT_DIR / "candidate_domain_equivalences.tsv"
DF_LOCI_PATH = OUT_DIR / "df_loci.pkl"
DF_LOCI_LABELED_PATH = OUT_DIR / "df_loci_labeled.pkl"
DF_ARCHITECTURES_PATH = OUT_DIR / "df_architectures.pkl"
DF_ARCHITECTURE_COUNTS_PATH = OUT_DIR / "df_architecture_counts.pkl"
ARCHITECTURE_FASTA_MANIFEST_PATH = OUT_DIR / "architecture_fastas.tsv"

REPRESENTATIVE_CLUSTER_MIN_IDENTITY = 0.95
REPRESENTATIVE_CLUSTER_COVERAGE = 0.95
MMSEQS_BIN = "mmseqs"
MMSEQS_THREADS = 8
REPRESENTATIVE_CLUSTERS_PATH = OUT_DIR / "representative_clusters.tsv"
REPRESENTATIVE_LIST_PATH = OUT_DIR / "representatives.tsv"
REPRESENTATIVE_FASTA_PATH = OUT_DIR / "representatives.faa"
MMSEQS_CLUSTER_TSV_PATH = OUT_DIR / "mmseqs_cluster.tsv"
REPRESENTATIVE_ALIGNMENTS_PATH = OUT_DIR / "representative_alignments.csv"
CLUSTER_SIMILARITY_PATH = OUT_DIR / "cluster_similarity.csv"

REPRESENTATIVE_CLUSTERING_BUNDLE = (REPRESENTATIVE_FASTA_PATH, REPRESENTATIVE_CLUSTERS_PATH,
                                    REPRESENTATIVE_LIST_PATH, MMSEQS_CLUSTER_TSV_PATH,
                                    REPRESENTATIVE_ALIGNMENTS_PATH, CLUSTER_SIMILARITY_PATH)

ANCHOR_FASTA_PATH = OUT_DIR / "anchors.faa"

AF3_DPAM_DIR = main_cfg.P_STAGE02 / "af3_dpam"
DPAM_DOMAINS_PATH = AF3_DPAM_DIR / "proteins_domains.csv"
DPAM_INPUT_FASTA_PATH = AF3_DPAM_DIR / "representatives.faa"
DPAM_ANCHOR_DOMAINS_PATH = AF3_DPAM_DIR / "proteins_domains_anchors.csv"
DPAM_ANCHOR_INPUT_FASTA_PATH = AF3_DPAM_DIR / "anchors.faa"

DPAM_BATCHES = ({"name": "representatives",
                 "domains_path": DPAM_DOMAINS_PATH,
                 "input_fasta": DPAM_INPUT_FASTA_PATH,
                 "description": "AF3+DPAM over the 2,287 MMseqs representatives."},
                {"name": "anchors",
                 "domains_path": DPAM_ANCHOR_DOMAINS_PATH,
                 "input_fasta": DPAM_ANCHOR_INPUT_FASTA_PATH,
                 "description": "AF3+DPAM over the ten experimental anchors' own sequences."})

DPAM_HTH_X_NAME = "HTH"
DPAM_TGROUP_ALIAS = {
    "328.8.1": "Schlafen",
    "225.2.1": "RecG",
    "225.1.1": "ATPase",
}

DPAM_OTHER_TOKEN = "OTHER"
DPAM_JUDGE_ORDER = ["good_domain", "partial_domain", "simple_topology", "low_confidence"]
DPAM_GOOD_JUDGE = "good_domain"
DPAM_HTH_CLASSES = ["no_HTH", "HTH+", "HTH++"]

DPAM_DOMAINS_TABLE_PATH = OUT_DIR / "dpam_domains.csv"
DPAM_ARCHITECTURES_PATH = OUT_DIR / "dpam_architectures.csv"
DPAM_PROTEIN_ARCHITECTURES_PATH = OUT_DIR / "dpam_protein_architectures.csv"
DPAM_ARCHITECTURE_OBSERVATIONS_PATH = OUT_DIR / "dpam_architecture_observations.csv"
DPAM_BATCH_MANIFEST_PATH = OUT_DIR / "dpam_batches.csv"

RESULT_ANCHOR_VALIDATION_PATH = OUT_DIR / "result_dpam_anchor_validation.csv"
RESULT_INTERPRO_VS_DPAM_HTH_PATH = OUT_DIR / "result_interpro_vs_dpam_hth.tsv"
RESULT_DPAM_JUDGE_SENSITIVITY_PATH = OUT_DIR / "result_dpam_judge_sensitivity.tsv"
RESULT_DPAM_FAMILY_CENSUS_PATH = OUT_DIR / "result_dpam_family_census.tsv"
