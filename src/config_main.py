from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
p_data = str(REPO_ROOT / "data") + "/"
p_data_db = p_data + "db/"
p_trial3 = p_data

p_trial1 = p_data + "Trial1/"

P_TRIAL3 = Path(p_trial3)
P_STAGE01 = P_TRIAL3 / "stage01_preprocessing"
P_STAGE02 = P_TRIAL3 / "stage02_domains"
P_STAGE03 = P_TRIAL3 / "stage03_concordance"
P_STAGE04 = P_TRIAL3 / "stage04_sae"

path_bacteria_assembly_summary = p_data_db + "Blast/refseq_assembly_summaries/bacteria/assembly_summary.txt"
path_archaea_assembly_summary = p_data_db + "Blast/refseq_assembly_summaries/archaea/assembly_summary.txt"

path_IPG = str(P_STAGE01 / "IPG") + "/"
path_sae_test = p_trial3 + "tests/sae_test/"

path_xml = str(P_TRIAL3 / "X4Y60Y89016-Alignment.xml")

path_mapped_0 = str(P_TRIAL3 / "mapped_0.csv")
path_addendum_csv = str(P_TRIAL3 / "addendum.csv")
path_mapped_1 = str(P_TRIAL3 / "mapped_1.csv")

path_interproscan_db = p_data_db + "interproscan-5.77-108.0/data"
p_interproscan = str(P_STAGE02 / "interproscan") + "/"
path_combined_fasta = p_interproscan + "input/combined.fasta"
path_addendum_fasta = p_interproscan + "input/addendum.fasta"
path_combined_addendum = p_interproscan + "input/combined_addendum.fasta"
fname = "combined_addendum.fasta"

path_interpro_tsv = p_interproscan + "output/combined_addendum.fasta.tsv"
path_pfam_clans_tsv = p_data_db + "PfamDB/pfam_clans.v38.0/Pfam-A.clans.tsv"

STUDY_SEQUENCE_FASTA = Path(path_combined_addendum)
STUDY_ROSTER_PATH = P_STAGE01 / "study_roster.tsv"

path_esmc_fasta = path_combined_addendum

DNA_BINDING_EVIDENCE_GRADES = ("direct_binding_and_repeat_necessity", "fragment_rescue",
                               "uninterpretable_assay", "untestable")

the_nine = {
    "NZ_CP196900.1": {"tRNAse_activity": 1, "DNA-binding_activity": 1, "DNA_binding_evidence": "direct_binding_and_repeat_necessity", "protein_acc": "WP_000554504.1", "Name":"StaN"},

    "NZ_CP018776.1": {"tRNAse_activity": 1, "DNA-binding_activity": 1, "DNA_binding_evidence": "fragment_rescue", "protein_acc": "WP_070704620.1", "Name":"ScoN"},
    "NZ_AP023389.1": {"tRNAse_activity": 1, "DNA-binding_activity": 1, "DNA_binding_evidence": "fragment_rescue", "protein_acc": "WP_023612761.1", "Name":"SpyN"},
    "NZ_CP020773.1": {"tRNAse_activity": 1, "DNA-binding_activity": 1, "DNA_binding_evidence": "fragment_rescue", "protein_acc": "WP_085236859.1", "Name":"SluN"},
    "NZ_CP115252.1": {"tRNAse_activity": 1, "DNA-binding_activity": 1, "DNA_binding_evidence": "fragment_rescue", "protein_acc": "WP_328984724.1", "Name":"PaeN"},
    "NZ_CP041983.1": {"tRNAse_activity": 1, "DNA-binding_activity": 2, "DNA_binding_evidence": "uninterpretable_assay", "protein_acc": "WP_265486636.1", "Name":"LacN"},
    "NZ_LR883000.1": {"tRNAse_activity": 0, "DNA-binding_activity": 2, "DNA_binding_evidence": "untestable", "protein_acc": "WP_074518199.1", "Name":"EcoN"},
    "NZ_OY764544.1": {"tRNAse_activity": 0, "DNA-binding_activity": 2, "DNA_binding_evidence": "untestable", "protein_acc": "WP_319539463.1", "Name":"MetN"},
    "NZ_CP096986.1": {"tRNAse_activity": 0, "DNA-binding_activity": 2, "DNA_binding_evidence": "untestable", "protein_acc": "WP_077836245.1", "Name":"CfeN"},
    "NZ_CP102268.1": {"tRNAse_activity": 0, "DNA-binding_activity": 2, "DNA_binding_evidence": "untestable", "protein_acc": "WP_006860708.1", "Name":"MfoN"},
}
