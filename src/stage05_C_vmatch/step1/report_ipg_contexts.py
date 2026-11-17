import csv
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from Bio import Entrez

from src.stage05_C_vmatch import config_vmatch as cfg

ASSEMBLY_SUMMARY_FIELDS = [
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
    "pubmed_id",
]

def resolve_ipg_path(protein_accession):
    upstream = cfg.IPG_DIR / f"{protein_accession}.xml"
    supplement = cfg.IPG_SUPPLEMENT_DIR / f"{protein_accession}.xml"
    if upstream.exists():
        return upstream
    if supplement.exists():
        return supplement
    return None

def missing_ipg_accessions(protein_accessions):
    return [
        accession
        for accession in sorted(set(protein_accessions))
        if resolve_ipg_path(accession) is None
    ]

def fetch_missing_ipgs(protein_accessions):
    missing = missing_ipg_accessions(protein_accessions)
    if not missing:
        print("No IPG reports are missing.")
        return []
    if not cfg.NCBI_EMAIL:
        raise ValueError("Set NCBI_EMAIL before fetching missing IPG reports.")

    cfg.IPG_SUPPLEMENT_DIR.mkdir(parents=True, exist_ok=True)
    Entrez.email = cfg.NCBI_EMAIL
    Entrez.tool = "repo_druid_stage05C_ipg_inventory"
    if cfg.NCBI_API_KEY:
        Entrez.api_key = cfg.NCBI_API_KEY

    written = []
    for number, accession in enumerate(missing, start=1):
        destination = cfg.IPG_SUPPLEMENT_DIR / f"{accession}.xml"
        with Entrez.efetch(
            db="protein",
            id=accession,
            rettype="ipg",
            retmode="xml",
        ) as handle:
            payload = handle.read()
        if isinstance(payload, bytes):
            payload = payload.decode("utf-8")

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=cfg.IPG_SUPPLEMENT_DIR,
                prefix=f".{accession}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            ET.parse(temporary)
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

        written.append(destination)
        print(f"[{number}/{len(missing)}] {destination}")
    return written

def load_assembly_pair_map(paths=None):
    pair_map = {}
    for path in paths or cfg.ASSEMBLY_SUMMARY_PATHS:
        with Path(path).open(newline="") as handle:
            rows = (
                line
                for line in handle
                if line.strip() and not line.startswith("#")
            )
            reader = csv.DictReader(
                rows,
                fieldnames=ASSEMBLY_SUMMARY_FIELDS,
                delimiter="\t",
            )
            for row in reader:
                assembly = row["assembly_accession"].strip()
                paired = row["gbrs_paired_asm"].strip()
                if not assembly:
                    continue
                canonical = _canonical_pair(assembly, paired)
                pair_map[assembly] = canonical
                if paired and paired != "na":
                    pair_map[paired] = canonical
    return pair_map

def _canonical_pair(assembly, paired):
    pair = [value for value in (assembly, paired) if value and value != "na"]
    refseq = [value for value in pair if value.startswith("GCF_")]
    return refseq[0] if refseq else sorted(pair)[0]

def parse_ipg_report(path, protein_accession):
    if path is None:
        return {
            "status": "missing",
            "path": "",
            "report_count": 0,
            "target_block_count": 0,
            "contexts": [],
        }

    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as error:
        return {
            "status": f"parse_error:{error}",
            "path": str(path),
            "report_count": 0,
            "target_block_count": 0,
            "contexts": [],
        }

    reports = root.findall("./IPGReport")
    contexts = []
    target_block_count = 0

    for report_index, report in enumerate(reports, start=1):
        product = report.find("./Product")
        product_accession = product.get("accver", "") if product is not None else ""
        product_name = product.get("name", "") if product is not None else ""

        for protein_index, protein in enumerate(
            report.findall("./ProteinList/Protein"),
            start=1,
        ):
            target_block = protein.get("accver", "") == protein_accession
            target_block_count += int(target_block)

            for cds_index, cds in enumerate(
                protein.findall("./CDSList/CDS"),
                start=1,
            ):
                context = {
                    "protein_accession": protein_accession,
                    "raw_context_id": (
                        f"{protein_accession}|ipg_{report_index:02d}|"
                        f"protein_{protein_index:04d}|cds_{cds_index:06d}"
                    ),
                    "context_source": "ipg",
                    "ipg_path": str(path),
                    "ipg_report_index": report_index,
                    "ipg_id": report.get("ipg", ""),
                    "product_accession": product_accession,
                    "product_name": product_name,
                    "member_protein_accession": protein.get("accver", ""),
                    "member_source": protein.get("source", ""),
                    "member_name": protein.get("name", ""),
                    "member_priority": protein.get("priority", ""),
                    "nucleotide_accession": cds.get("accver", ""),
                    "start": _integer(cds.get("start")),
                    "stop": _integer(cds.get("stop")),
                    "strand": cds.get("strand", ""),
                    "assembly": cds.get("assembly", ""),
                    "taxid": cds.get("taxid", ""),
                    "organism": cds.get("org", ""),
                    "strain": cds.get("strain", ""),
                    "target_protein_block": target_block,
                    "refseq_member": protein.get("source", "") == "RefSeq",
                    "refseq_nucleotide": _is_refseq_nucleotide(
                        cds.get("accver", "")
                    ),
                    "authoritative_context": False,
                }
                context["has_valid_interval"] = _valid_interval(context)
                context["has_assembly"] = bool(context["assembly"])
                contexts.append(context)

    status = "ok"
    if not reports:
        status = "no_ipg_report"
    elif not target_block_count:
        status = "target_block_missing"

    return {
        "status": status,
        "path": str(path),
        "report_count": len(reports),
        "target_block_count": target_block_count,
        "contexts": contexts,
    }

def _integer(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

def _valid_interval(context):
    return (
        bool(context["nucleotide_accession"])
        and context["start"] is not None
        and context["stop"] is not None
        and 1 <= context["start"] <= context["stop"]
        and context["strand"] in {"+", "-"}
    )

def _is_refseq_nucleotide(accession):
    return accession.startswith(("NC_", "NG_", "NM_", "NR_", "NT_", "NW_", "NZ_"))
