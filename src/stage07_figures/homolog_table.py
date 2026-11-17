from pathlib import Path

import pandas as pd

from src import roster
from src import config_main as main_cfg
from src.protein_tree import config_tree as tree_cfg
from src.stage02_domains import config_domains as domains_cfg
from src.stage07_figures import config_figures as cfg
from src.stage07_figures.tree.prepare import phylum_names, sample_phyla

MAPPED_COLUMNS = {"source_name": "Protein accession", "org": "Organism", "taxid": "Taxonomy ID", "kingdom": "Domain", "assembly": "Assembly",
                  "blast_e value": "BLASTp E-value", "blast_pident": "BLASTp identity (%)",
                  "blast_qcov": "BLASTp query coverage (fraction)"}
ARCHITECTURE_COLUMNS = {"protein_accession": "Protein accession", "cluster_id": "Cluster", "representative_accession": "Cluster representative",
                        "cluster_size": "Cluster size", "hth_families": "HTH families"}
STATUS_COLUMNS = {"protein_accession": "Protein accession", "status": "Status", "reasons": "Reasons", "hth_class": "HTH class",
                  "architecture": "Domain architecture", "length": "Length (aa)"}
SHOWN = ("Protein accession", "Lab Tested", "Organism", "Taxonomy ID", "Domain", "Phylum", "Cluster host phylum", "Assembly", "BLASTp E-value",
         "BLASTp identity (%)", "BLASTp query coverage (fraction)", "Cluster", "Cluster representative", "Cluster size", "Status", "Reasons",
         "HTH class", "HTH families", "Domain architecture", "Length (aa)")

CLOSING = ("Some proteins have no genome record at NCBI, and a few organisms have no phylum named in NCBI Taxonomy. "
           "Cells that would describe a genome or a phylum are left empty for those proteins.")

NOTES = {
    "Protein accession": "NCBI RefSeq protein accession.",
    "Lab Tested": "TRUE for the ten proteins tested experimentally.",
    "Organism": "Organism of a genome that encodes the protein, linked through NCBI Identical Protein Groups. The same protein can occur in many genomes.",
    "Taxonomy ID": "NCBI Taxonomy ID of that organism.",
    "Domain": "Domain of life of that organism: Bacteria or Archaea.",
    "Phylum": "Phylum of that organism in NCBI Taxonomy.",
    "HTH families": "The ECOD family of each HTH domain, in order, separated by semicolons. "
                    "A name written as \"T:\" followed by a description is ECOD's topology group, used where ECOD names no family for that domain.",
    "Cluster host phylum": "Most common phylum over all genomes that carry the intact proteins of this protein's cluster. The phylum counts per "
                           "cluster in the text are counted from this column. Intact proteins only.",
    "Assembly": "NCBI assembly accession of the reference genome.",
    "BLASTp E-value": "E-value of the protein's best BLASTp alignment to StaN.",
    "BLASTp identity (%)": "Percent identity of that alignment.",
    "BLASTp query coverage (fraction)": "Fraction of StaN covered by that alignment.",
    "Cluster": f"MMseqs2 cluster of proteins sharing at least {domains_cfg.REPRESENTATIVE_CLUSTER_MIN_IDENTITY:.0%} identity over "
               f"{domains_cfg.REPRESENTATIVE_CLUSTER_COVERAGE:.0%} coverage.",
    "Cluster representative": "The cluster member whose AlphaFold 3 structure was parsed into domains (DPAM, ECOD); its domain architecture "
                              "is given to every member of the cluster.",
    "Cluster size": "Number of proteins in the cluster.",
    "Status": "intact: starts with methionine, carries complete Schlafen and RecG domains, and is not an exact piece of a longer protein in "
              "this table. suspect: one of these fails. unresolved: a core domain is between truncated and complete, or no domain call "
              "exists. Only intact proteins enter the analyses.",
    "Reasons": f"Why a protein is not intact, separated by semicolons. no_start_m: does not start with methionine. <domain>_absent: the "
               f"domain is missing. <domain>_truncated: the domain is shorter than {tree_cfg.CORE_TRUNCATED_SHARE:.0%} of the shortest "
               f"experimentally tested protein's. <domain>_short: between {tree_cfg.CORE_TRUNCATED_SHARE:.0%} and "
               f"{tree_cfg.CORE_COMPLETE_SHARE:.0%}. piece_lacking_<domains>: an exact piece of a longer protein in this table that "
               f"carries those domains. no_domain_call: its structure has no domain call. Empty for intact proteins.",
    "HTH class": "no_HTH: no helix-turn-helix (HTH) domain. HTH+: one. HTH++: two or more.",
    "Domain architecture": "Domains from N- to C-terminus, from the cluster representative's structure.",
    "Length (aa)": "Protein length in amino acids."}

CORRECTION_PATH = Path(__file__).with_name("last_minute_correction_to_anchors.tsv")
CORRECTED_COLUMNS = {"organism": "Organism", "taxonomy_id": "Taxonomy ID", "domain": "Domain", "phylum": "Phylum"}

def corrected(mapped):

    patch = pd.read_csv(CORRECTION_PATH, sep="\t", comment="#", dtype=str).set_index("protein_accession")
    patch = patch.rename(columns=CORRECTED_COLUMNS)

    missing = set(patch.index) - set(mapped["Protein accession"])

    if missing: raise ValueError(f"{CORRECTION_PATH.name} names proteins Table S4 does not hold: {sorted(missing)}")

    rows = mapped["Protein accession"].isin(patch.index)
    place = mapped.loc[rows, "Protein accession"]

    for column in CORRECTED_COLUMNS.values(): mapped.loc[rows, column] = place.map(patch[column]).to_numpy()

    mapped["Taxonomy ID"] = mapped["Taxonomy ID"].astype("Int64")

    return mapped

def table():

    mapped = pd.read_csv(main_cfg.path_mapped_1, usecols=list(MAPPED_COLUMNS)).rename(columns=MAPPED_COLUMNS)
    architectures = pd.read_csv(domains_cfg.DPAM_PROTEIN_ARCHITECTURES_PATH, usecols=list(ARCHITECTURE_COLUMNS)).rename(columns=ARCHITECTURE_COLUMNS)

    architectures["HTH families"] = architectures["HTH families"].str.replace(" + ", "; ", regex=False)
    status = pd.read_csv(tree_cfg.PROTEIN_STATUS_PATH, sep="\t", usecols=list(STATUS_COLUMNS)).rename(columns=STATUS_COLUMNS)

    mapped["Taxonomy ID"] = mapped["Taxonomy ID"].astype("Int64")
    taxids = mapped["Taxonomy ID"].astype("string")
    mapped["Phylum"] = taxids.map(phylum_names(set(taxids.dropna())))

    mapped = corrected(mapped)

    samples = pd.read_csv(tree_cfg.SAMPLES_PATH, sep="\t", dtype=str)
    samples["Cluster host phylum"] = samples["sample_accession"].map(sample_phyla())
    samples = samples.rename(columns={"protein_accession": "Protein accession"}).drop(columns="sample_accession")

    result = roster.read()[["protein_accession"]].rename(columns={"protein_accession": "Protein accession"})

    for source in (mapped, samples, architectures, status): result = result.merge(source, on="Protein accession", how="left", validate="one_to_one")

    result["Lab Tested"] = result["Protein accession"].isin({anchor["protein_acc"] for anchor in main_cfg.the_nine.values()})

    return result[list(SHOWN)]

def draw():

    result = table()
    csv_path = cfg.HOMOLOG_TABLE_DIR / "table_S4_homologs.csv"
    excel_path = cfg.HOMOLOG_TABLE_DIR / "table_S4_homologs.xlsx"
    notes = pd.DataFrame({"Column": list(SHOWN) + ["", "Blank cells"], "Description": [NOTES[column] for column in SHOWN] + ["", CLOSING]})
    unlisted = result["Organism"].isna()

    cfg.HOMOLOG_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        notes.to_excel(writer, sheet_name="Columns", index=False)
        result.to_excel(writer, sheet_name="Table S4", index=False)
        writer.sheets["Columns"].column_dimensions["A"].width = 32
        writer.sheets["Columns"].column_dimensions["B"].width = 120
        writer.sheets["Table S4"].freeze_panes = "B2"

    print(f"Table S4: {len(result):,} proteins; {unlisted.sum():,} without BLASTp/genome row, of which "
          f"{(unlisted & (result['Status'] == 'intact')).sum():,} intact")

    return [csv_path, excel_path]
