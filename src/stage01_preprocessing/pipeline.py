import argparse
import os
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import pandas as pd
from Bio import Entrez, SeqIO
from Bio.Blast import NCBIXML
from Bio.Blast import Record as blastRecord
from dotenv import load_dotenv
from tqdm import tqdm

from src import config_main as cfg

ASSEMBLY_FIELDS = [
    "assembly_level",
    "group",
    "total_gene_count",
    "genome_size",
    "gc_percent",
    "replicon_count"]

def configure_entrez(email, api_key=None):

    if not email:

        raise ValueError("Set NCBI_EMAIL before downloading IPG reports.")

    Entrez.email = email

    if api_key:

        Entrez.api_key = api_key

class ReadBlastOutputXML:

    def __init__(self, path):

        self.path = path
        self.path_csv = path.replace(".xml", ".csv")

        with open(path) as handle:

            self.blast_record: blastRecord = NCBIXML.read(handle)

        m = re.search(r'/([A-Z0-9]+)-Alignment.xml', path)
        self.ridnum = m.group(1)

    def get_blast_hit_dataframe(self):

        bh_dicts = []

        for alignment in self.blast_record.alignments:

            ss          = alignment.title.split('|')
            accession   = ss[1]
            description = ss[2]

            for hsp in alignment.hsps:

                x = hsp.query_end - hsp.query_start + 1

                d = {'Accession': accession,
                     'RID': self.ridnum,
                     'Frame': hsp.frame[1],
                     'Description': description.strip(),
                     'Start': hsp.sbjct_start,
                     'End': hsp.sbjct_end,
                     'E Value': hsp.expect,
                     'Bits': hsp.bits,
                     'Score': hsp.score,
                     'QCov': x/self.blast_record.query_length,
                     'Pident': 100.0*hsp.identities/hsp.align_length}

                bh_dicts.append(d)

        return pd.DataFrame(bh_dicts)

    def export2csv(self):

        if os.path.exists(self.path_csv):

            return pd.read_csv(self.path_csv)

        df = self.get_blast_hit_dataframe()
        df.to_csv(self.path_csv, index=False)

        return df

class IdenticalProteinGroup:

    def __init__(self, protID, path_IPG):

        self.protID   = protID
        self.path_IPG = path_IPG
        self.path_xml = os.path.join(self.path_IPG, f"{self.protID}.xml")
        self.path_csv = os.path.join(self.path_IPG, f"{self.protID}.csv")
        self.path_csv_failed = os.path.join(self.path_IPG, f"{self.protID}_failed.csv")

    def fetch(self):

        id_list = [self.protID]

        if os.path.exists(self.path_xml):

            return False

        with Entrez.efetch(db="protein", id=id_list, rettype="ipg", retmode="xml") as handle:

            text = handle.read().decode("utf-8")

            with open(self.path_xml, "w") as output:

                output.write(text)

        return True

    def parse_xml(self, limit=None):

        root = ET.parse(self.path_xml).getroot()

        if root.tag != "IPGReportSet":

            raise ValueError(f"expected an IPGReportSet response, got <{root.tag}>")

        protein = root.find(".//Protein")

        if protein is None: return pd.DataFrame()

        cds_list = [cds.attrib for cds in protein.findall("./CDSList/CDS")]

        if limit is not None:

            cds_list = cds_list[: int(limit)]

        return pd.DataFrame(cds_list)

    def download_and_process(self, assembly_lookup, fields_to_add):

        if os.path.exists(self.path_csv):

            return "cached"

        self.fetch()

        try:

            df_ipg = self.parse_xml(limit=None)

        except (ET.ParseError, OSError, ValueError) as exc:

            self._write_failure("parse_error", f"{type(exc).__name__}: {exc}")

            return "failed"

        if df_ipg.empty:

            self._write_failure("empty_ipg_report", "NCBI reported no identical-protein-group members.")

            return "absent"

        if "assembly" not in df_ipg.columns: df_ipg["assembly"] = ""

        df_ipg = df_ipg.join(assembly_lookup[fields_to_add], on="assembly")
        df_ipg.to_csv(self.path_csv, index=False)

        if os.path.exists(self.path_csv_failed): os.remove(self.path_csv_failed)

        return "written"

    def _write_failure(self, outcome, detail):

        pd.DataFrame([{"protein_accession": self.protID, "outcome": outcome, "detail": detail}]).to_csv(self.path_csv_failed, index=False)

class AssemblySummary:

    def __init__(self, path_assembly_summary):

        self.path_assembly_summary = path_assembly_summary

        cols = [
            "assembly_accession",
            "bioproject",
            "biosample",
            "wgs_master",
            "refseq_category",
            "taxid",
            "species_taxid",
            "organism_name",
            "infraspecific_name",
            "isolate",
            "version_status",
            "assembly_level",
            "release_type",
            "genome_rep",
            "seq_rel_date",
            "asm_name",
            "asm_submitter",
            "gbrs_paired_asm",
            "paired_asm_comp",
            "ftp_path",
            "excluded_from_refseq",
            "relation_to_type_material",
            "asm_not_live_date",
            "assembly_type",
            "group",
            "genome_size",
            "genome_size_ungapped",
            "gc_percent",
            "replicon_count",
            "scaffold_count",
            "contig_count",
            "annotation_provider",
            "annotation_name",
            "annotation_date",
            "total_gene_count",
            "protein_coding_gene_count",
            "non_coding_gene_count",
            "pubmed_id"]

        df = pd.read_csv(self.path_assembly_summary, sep='\t', comment='#', names=cols)

        self.df = df

def collect_complete_genomes(paths):

    collected = []

    for path in paths:

        df = pd.read_csv(path)
        filename = os.path.basename(path)
        complete = df.loc[df["assembly_level"] == "Complete Genome"]
        source_name = os.path.splitext(os.path.basename(path))[0]

        if not complete.empty:

            selected = complete.iloc[[0]].copy()
            selected["source_name"] = source_name
            collected.append(selected)
            continue

        if df.empty:

            continue

        genome_size_numeric = pd.to_numeric(df["genome_size"], errors="coerce")

        if genome_size_numeric.notna().any():

            idx = genome_size_numeric.idxmax()
            selected = df.loc[[idx]].copy()

        else:

            selected = df.iloc[[0]].copy()

        selected["source_name"] = source_name
        collected.append(selected)

    if collected:

        return pd.concat(collected, ignore_index=True)

    return pd.DataFrame()

def select_best_blast_hsps(df_blast):

    hsp_counts = df_blast["Accession"].value_counts()
    selected = (df_blast.assign(_row_order=range(len(df_blast)))
                .sort_values(["Accession", "E Value", "Bits", "QCov", "Pident", "_row_order"],
                             ascending=[True, True, False, False, False, True],)
                .drop_duplicates("Accession")
                .drop(columns="_row_order"))
    selected["HSP Count"] = selected["Accession"].map(hsp_counts)
    selected = selected.rename(columns={column: "blast_" + column.lower() for column in selected.columns})

    return selected.rename(columns={"blast_hsp count": "blast_hsp_count"})

def build_mapped_0(ipg_paths, df_blast):

    contexts = collect_complete_genomes(ipg_paths)
    blast_hsps = select_best_blast_hsps(df_blast)

    mapped = contexts.merge(blast_hsps, how="left", left_on="source_name",
                            right_on="blast_accession", validate="one_to_one")
    mapped["in_blast"] = True
    mapped["is_addendum"] = False
    mapped["context_source"] = "ipg"

    return mapped

def build_mapped_1(df_mapped_0, df_addendum, df_blast):

    if df_addendum["blast_accession"].duplicated().any():

        raise ValueError("addendum.csv contains duplicate protein accessions")

    blast_hsp_counts = df_blast["Accession"].value_counts()
    addendum = df_addendum.copy()
    addendum["in_blast"] = addendum["blast_accession"].isin(blast_hsp_counts.index)
    addendum["blast_hsp_count"] = addendum["blast_accession"].map(blast_hsp_counts).fillna(0).astype(int)
    addendum["is_addendum"] = True
    addendum["context_source"] = "addendum"

    addendum_accessions = set(addendum["blast_accession"])
    mapped = df_mapped_0[~df_mapped_0["blast_accession"].isin(addendum_accessions)]
    mapped = pd.concat([mapped, addendum], ignore_index=True)

    if mapped["blast_accession"].duplicated().any():

        raise ValueError("mapped_1 must contain one row per protein accession")

    return mapped

def step1_download_ipgs():

    load_dotenv()
    configure_entrez(os.environ.get("NCBI_EMAIL"), os.environ.get("NCBI_API_KEY"),)
    Path(cfg.path_IPG).mkdir(parents=True, exist_ok=True)

    assemblies = pd.concat([AssemblySummary(cfg.path_bacteria_assembly_summary).df,
                            AssemblySummary(cfg.path_archaea_assembly_summary).df],
                           ignore_index=True)
    assembly_lookup = assemblies.set_index("assembly_accession")[ASSEMBLY_FIELDS]
    accessions = ReadBlastOutputXML(cfg.path_xml).export2csv()["Accession"].drop_duplicates()
    statuses = Counter()

    for accession in tqdm(accessions, desc="Stage 01 IPG", unit="protein"):

        IPG = IdenticalProteinGroup(accession, cfg.path_IPG,)

        status = IPG.download_and_process(assembly_lookup, ASSEMBLY_FIELDS,)
        statuses[status] += 1

    summary = ", ".join(f"{key}={value:,}" for key, value in sorted(statuses.items()))

    print(f"Stage 01 IPG reports: {len(accessions):,} proteins; {summary}")
    print(cfg.path_IPG)

def step2_build_mappings():

    blast = ReadBlastOutputXML(cfg.path_xml).export2csv()
    ipg_paths = [Path(cfg.path_IPG) / f"{accession}.csv"
                 for accession in blast["Accession"].drop_duplicates()
                 if (Path(cfg.path_IPG) / f"{accession}.csv").exists()]

    mapped_0 = build_mapped_0(ipg_paths, blast)
    mapped_0_path = Path(cfg.path_mapped_0)
    mapped_0_path.parent.mkdir(parents=True, exist_ok=True)
    mapped_0.to_csv(mapped_0_path, index=False)

    addendum = pd.read_csv(cfg.path_addendum_csv)
    mapped_1 = build_mapped_1(mapped_0, addendum, blast)
    mapped_1_path = Path(cfg.path_mapped_1)
    mapped_1.to_csv(mapped_1_path, index=False)

    print(f"Stage 01 mappings: mapped_0={len(mapped_0):,} rows; mapped_1={len(mapped_1):,} rows")
    print(mapped_0_path)
    print(mapped_1_path)

def step3_build_fasta():

    output_path = Path(cfg.path_combined_addendum)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen = set()
    written = 0

    with output_path.open("w") as output:

        for input_path in (cfg.path_combined_fasta, cfg.path_addendum_fasta):

            for record in SeqIO.parse(input_path, "fasta"):

                if record.id in seen:

                    continue

                SeqIO.write(record, output, "fasta")
                seen.add(record.id)
                written += 1

    print(f"Stage 01 combined FASTA: {written:,} unique proteins")
    print(output_path)

def run():

    step1_download_ipgs()
    step2_build_mappings()
    step3_build_fasta()

def main(argv=None):

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p1 = sub.add_parser("step1_download_ipgs", help="Download or resume per-protein IPG reports.")
    p1.set_defaults(func=step1_download_ipgs)

    p2 = sub.add_parser("step2_build_mappings", help="Build mapped_0.csv and mapped_1.csv.")
    p2.set_defaults(func=step2_build_mappings)

    p3 = sub.add_parser("step3_build_fasta", help="Build the deduplicated combined FASTA.")
    p3.set_defaults(func=step3_build_fasta)

    p_run = sub.add_parser("run", help="Run the complete Stage 01 pipeline.")
    p_run.set_defaults(func=run)

    args = parser.parse_args(argv)
    args.func()

if __name__ == "__main__":

    main()
