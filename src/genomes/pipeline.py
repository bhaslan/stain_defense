import argparse
import json
import shutil
import subprocess
import zipfile

import pandas as pd

from src.genomes import config_genomes as cfg

def assemblies():

    units = pd.read_csv(cfg.UNIT_MEASURES_PATH, sep="\t", dtype=str, usecols=["protein_accession", "source_context_id"]).drop_duplicates()
    contexts = pd.read_csv(cfg.CONTEXTS_PATH, sep="\t", dtype=str, usecols=["protein_accession", "context_id", "canonical_assembly", "assembly"])
    merged = units.merge(contexts, left_on=["protein_accession", "source_context_id"], right_on=["protein_accession", "context_id"], how="left")

    if missing := merged["context_id"].isna().sum(): raise RuntimeError(f"{cfg.CONTEXTS_PATH.name} lacks {missing:,} contexts of Step 6C's flanks.")

    return sorted(merged["canonical_assembly"].fillna(merged["assembly"]).unique())

def _done(accession): return (cfg.GENOMES_DIR / accession / cfg.REPORT_NAME).exists()

def _fetch(batch):

    shutil.rmtree(cfg.STAGING_DIR, ignore_errors=True)
    cfg.STAGING_DIR.mkdir(parents=True)

    accessions = cfg.STAGING_DIR / "accessions.txt"
    accessions.write_text("\n".join(batch) + "\n")
    package = cfg.STAGING_DIR / "package.zip"

    subprocess.run(["datasets", "download", "genome", "accession", "--inputfile", str(accessions),
                    "--include", ",".join(cfg.DATASETS_INCLUDE), "--filename", str(package), "--no-progressbar"], check=True)

    with zipfile.ZipFile(package) as archive: archive.extractall(cfg.STAGING_DIR)

    data = cfg.STAGING_DIR / "ncbi_dataset" / "data"
    reports = [json.loads(line) for line in (data / "assembly_data_report.jsonl").read_text().splitlines()]

    for report in reports:

        accession = report["accession"]
        source = data / accession
        target = cfg.GENOMES_DIR / accession
        target.mkdir(exist_ok=True)

        for name, pattern in cfg.FILES.values():

            found = list(source.glob(pattern))
            if len(found) > 1: raise RuntimeError(f"{accession}: {len(found)} files match {pattern}")
            if found: shutil.move(found[0], target / name)

        (target / cfg.REPORT_NAME).write_text(json.dumps(report) + "\n")

    shutil.rmtree(cfg.STAGING_DIR)

def manifest(wanted):

    rows = []

    for accession in wanted:

        folder = cfg.GENOMES_DIR / accession
        row = {"accession": accession, "downloaded": _done(accession)}
        row.update({f"has_{label}": (folder / name).exists() for label, (name, _) in cfg.FILES.items()})

        if row["downloaded"]:

            report = json.loads((folder / cfg.REPORT_NAME).read_text())
            stats = report.get("assemblyStats", {})
            row.update(organism=report.get("organism", {}).get("organismName"), gc_percent=stats.get("gcPercent"),
                       total_sequence_length=stats.get("totalSequenceLength"), contigs=stats.get("numberOfContigs"),
                       annotation_provider=report.get("annotationInfo", {}).get("provider"))

        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(cfg.MANIFEST_PATH, sep="\t", index=False)

    return table

def download(limit=None):

    wanted = assemblies()
    missing = [accession for accession in wanted if not _done(accession)]
    todo = missing[:limit]
    cfg.GENOMES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Genomes: {len(wanted):,} assemblies, {len(wanted) - len(missing):,} already downloaded, {len(todo):,} to fetch")

    for start in range(0, len(todo), cfg.BATCH_SIZE):

        _fetch(todo[start:start + cfg.BATCH_SIZE])
        print(f"Genomes: {min(start + cfg.BATCH_SIZE, len(todo)):,}/{len(todo):,} fetched", flush=True)

    table = manifest(wanted)

    print(f"Genomes: {table['downloaded'].sum():,} of {len(table):,} downloaded; {table['has_gff'].sum():,} with annotation, {table['has_faa'].sum():,} with proteins")
    print(cfg.MANIFEST_PATH)

def main():

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    download_parser = sub.add_parser("download")
    download_parser.add_argument("--limit", type=int, default=None, help="fetch at most this many (a pilot)")
    download_parser.set_defaults(func=lambda args: download(args.limit))

    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":

    main()
