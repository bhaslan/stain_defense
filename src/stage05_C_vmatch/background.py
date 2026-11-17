import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import AfterPosition, BeforePosition, ExactPosition, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

from src import roster
from src.genomes import config_genomes as genomes_cfg
from src.genomes.assembly import Assembly
from src.stage05_C_vmatch import config_vmatch as cfg
from src.stage05_C_vmatch.step2 import download_extract, segment_downloader
from src.stage05_C_vmatch.step3c import trf, vmatch
from src.stage05_C_vmatch.step3c.trf_evidence import TRF_ARRAY_FIELDS
from src.stage05_C_vmatch.step4c import pipeline as step4c
from src.stage05_C_vmatch.step4c.evidence import build_composition_rows, build_evidence_items
from src.stage05_C_vmatch.step5c.measures import counts_evidence, searched_unit_rows

BOUNDARY_FIELDS = tuple(f"{side}_{suffix}" for side in ("left", "right")
                        for suffix in ("boundary_status", "intergenic_length", "neighbor_cds_start", "neighbor_cds_stop",
                                       "neighbor_cds_strand", "neighbor_boundary_type", "neighbor_partial"))

def focal_units():

    units = pd.read_csv(cfg.STEP6_UNIT_MEASURES_PATH, sep="\t", dtype=str)
    units = units.loc[units["repeat_set"] == cfg.STEP5_PRIMARY_REPEAT_SET,
                      ["group_accession", "protein_accession", "side", "search_unit_id", "source_context_id"]]
    contexts = pd.read_csv(cfg.STEP2_CONTEXTS_PATH, sep="\t", dtype=str, keep_default_na=False)
    merged = units.merge(contexts, left_on=["protein_accession", "source_context_id"], right_on=["protein_accession", "context_id"],
                         how="left", validate="many_to_one")

    if missing := (merged["context_id"] == "").sum() + merged["context_id"].isna().sum():

        raise ValueError(f"{cfg.STEP2_CONTEXTS_PATH.name} lacks {missing:,} contexts of kept Step 6C units")

    merged["genome"] = merged["canonical_assembly"].where(merged["canonical_assembly"] != "", merged["assembly"])

    return merged

def contributing_genomes(assemblies="GCF_"):

    manifest = pd.read_csv(genomes_cfg.MANIFEST_PATH, sep="\t")
    genomes = set(focal_units()["genome"])
    manifest = manifest.loc[manifest["accession"].isin(genomes) & manifest["accession"].str.startswith(assemblies)
                            & manifest["has_gff"] & manifest["has_faa"]].copy()
    manifest["genus"] = manifest["organism"].map(_genus)

    return manifest.sort_values("accession").reset_index(drop=True)

def pilot_genomes():

    genomes = contributing_genomes()
    rng = np.random.default_rng(cfg.BACKGROUND_SEED)
    shuffled = genomes.iloc[rng.permutation(len(genomes))]

    return sorted(shuffled.drop_duplicates("genus")["accession"].head(cfg.BACKGROUND_PILOT_GENOMES))

def _genus(organism):

    words = str(organism).split()

    return " ".join(words[:2]) if words and words[0] == "Candidatus" else (words[0] if words else "")

def genbank_interval(assembly, context):

    fetch_start, fetch_stop = segment_downloader.fetch_bounds(context, cfg.FLANK_LENGTH)
    name = assembly.names[context["nucleotide_accession"]]
    sequence = assembly.sequences[name]
    stop = min(fetch_stop, len(sequence))
    length = stop - fetch_start + 1
    cds = assembly.cds
    near = cds.loc[(cds["seqid"] == name) & (cds["end"] >= fetch_start) & (cds["start"] <= stop)]
    features = []

    for row in near.itertuples():

        left = (BeforePosition(0) if row.start < fetch_start
                else BeforePosition(row.start - fetch_start) if row.start_fuzzy else ExactPosition(row.start - fetch_start))
        right = (AfterPosition(length) if row.end > stop
                 else AfterPosition(row.end - fetch_start + 1) if row.end_fuzzy else ExactPosition(row.end - fetch_start + 1))
        features.append(SeqFeature(FeatureLocation(left, right, strand={"+": 1, "-": -1}.get(row.strand)), type="CDS"))

    return SeqRecord(Seq(sequence[fetch_start - 1:stop]), id=name, features=features)

def focal_check(genomes):

    units = focal_units()
    units = units.loc[units["genome"].isin(genomes)]
    sequences = roster.sequences()
    stored = {}
    rows = []

    for genome, group in units.groupby("genome"):

        assembly = Assembly(genome)
        cds = assembly.cds

        for unit in group.to_dict("records"):

            row = {key: unit[key] for key in ("group_accession", "protein_accession", "side", "search_unit_id", "source_context_id", "genome")}
            row.update(_cds_flags(assembly, cds, unit))

            if unit["nucleotide_accession"] not in assembly.names:

                rows.append({**row, "status": "sequence_absent"})
                continue

            cut, records = download_extract.process_record(dict(unit), genbank_interval(assembly, unit), unit, sequences[unit["protein_accession"]])
            flank = str(records[unit["side"]].seq) if records else ""
            side_path = download_extract._protein_paths(unit["protein_accession"])[f"validated_{unit['side']}"]
            if side_path not in stored: stored[side_path] = {record.id: str(record.seq) for record in SeqIO.parse(side_path, "fasta")}
            previous = stored[side_path].get(unit["sequence_id"])
            rows.append({**row, "status": cut["status"], "cut_matches": previous is not None and flank == previous,
                         "boundaries_match": all(str(cut.get(field, "")) == str(unit[field]) for field in BOUNDARY_FIELDS)})

    return pd.DataFrame(rows)

def _cds_flags(assembly, cds, unit):

    name = assembly.names.get(unit["nucleotide_accession"])
    match = cds.loc[(cds["seqid"] == name) & (cds["start"] == int(unit["start"])) & (cds["end"] == int(unit["stop"]))
                    & (cds["strand"] == unit["strand"])]

    if match.empty: return {"cds_found": False, "cds_pseudo": "", "cds_partial": ""}

    first = match.iloc[0]

    return {"cds_found": True, "cds_pseudo": bool(first["pseudo"]),
            "cds_partial": bool(first["partial"] or first["start_fuzzy"] or first["end_fuzzy"])}

def panel_dir(panel): return cfg.BACKGROUND_DIR / panel

def panel_genomes(panel):

    if panel == "pilot": return pilot_genomes()

    return contributing_genomes(cfg.BACKGROUND_PANELS[panel]["assemblies"])["accession"].tolist()

def genome_rng(genome):

    digest = int(hashlib.sha256(genome.encode()).hexdigest()[:16], 16)

    return np.random.default_rng([cfg.BACKGROUND_SEED, digest])

def gene_exclusions(assembly, study_regions, roster_ids):

    cds = assembly.cds
    reach = cfg.FLANK_LENGTH
    near_study = np.zeros(len(cds), dtype=bool)

    for seqid, start, stop in study_regions:

        near_study |= ((cds["seqid"] == seqid) & (cds["start"] - reach <= stop + reach) & (cds["end"] + reach >= start - reach)).to_numpy()

    rules = [("no_protein", cds["protein_id"] == ""), ("pseudogene", cds["pseudo"]),
             ("partial", cds["partial"] | cds["start_fuzzy"] | cds["end_fuzzy"]), ("several_parts", cds["parts"] > 1),
             ("study_protein", cds["protein_id"].isin(roster_ids)), ("near_study_context", near_study),
             ("no_protein_sequence", ~cds["protein_id"].isin(list(assembly.proteins)))]

    return pd.Series(np.select([np.asarray(rule, dtype=bool) for _, rule in rules], [name for name, _ in rules], default=""), index=cds.index)

def drawn_genes(assembly, exclusions, count):

    eligible = assembly.cds.loc[exclusions == ""]
    order = genome_rng(assembly.accession).permutation(len(eligible))

    return eligible.iloc[order[:count]]

def cut_gene(assembly, gene):

    sequence_id = f"{assembly.accession}__{gene.cds_id}"
    context = {"protein_accession": sequence_id, "context_id": gene.cds_id, "sequence_id": sequence_id,
               "canonical_assembly": assembly.accession, "nucleotide_accession": gene.seqid, "start": gene.start,
               "stop": gene.end, "strand": gene.strand, "authoritative_context": True, "protein_id": gene.protein_id}

    return download_extract.process_record(dict(context), genbank_interval(assembly, context), context, assembly.proteins[gene.protein_id])

def search_units(row, records):

    units = []

    for side in ("left", "right"):

        if row[f"{side}_length"] < cfg.MIN_FLANK_LENGTH or not set(str(records[side].seq)) <= set("ACGT"): continue
        other = "right" if side == "left" else "left"
        units.append({"protein_accession": row["sequence_id"], "candidate_group": "background", "experimental_anchor": "",
                      "scope": f"within_{side}", "search_unit_id": "background", "source_context_id": row["context_id"],
                      "source_sequence_id": row["sequence_id"], "selection_rank": 1, "authoritative_context": True,
                      "left_sequence_eligible": row["left_length"] >= cfg.MIN_FLANK_LENGTH,
                      "right_sequence_eligible": row["right_length"] >= cfg.MIN_FLANK_LENGTH,
                      f"{side}_cluster_id": "background", f"{side}_cluster_size": 1,
                      f"{side}_cluster_representative_sequence_id": row["sequence_id"],
                      f"{other}_cluster_id": "", f"{other}_cluster_size": "", f"{other}_cluster_representative_sequence_id": "",
                      "paired_cluster_id": "", "pair_support": "", "database_side": side, "query_side": "",
                      "database_sequence_id": row["sequence_id"], "database_sequence": str(records[side].seq),
                      "query_sequence_id": "", "query_sequence": "", "context": row})

    return units

def study_regions(assembly, contexts):

    rows = contexts.loc[contexts["genome"] == assembly.accession]

    return [(assembly.names[name], int(start), int(stop)) for name, start, stop in zip(rows["nucleotide_accession"], rows["start"], rows["stop"])
            if name in assembly.names]

GENE_FIELDS = ("genome", "sequence_id", "cds_id", "protein_id", "seqid", "start", "stop", "strand", "status",
               "left_length", "right_length", "left_gc", "right_gc")

def _search_genome(task):

    panel, genome, count, contexts, roster_ids = task
    assembly = Assembly(genome)
    exclusions = gene_exclusions(assembly, study_regions(assembly, contexts), roster_ids)
    genes, units, flank_sequences = [], [], []

    for gene in drawn_genes(assembly, exclusions, count).itertuples():

        row, records = cut_gene(assembly, gene)
        genes.append({"genome": genome, "sequence_id": row["sequence_id"], "cds_id": gene.cds_id, "protein_id": gene.protein_id,
                      "seqid": gene.seqid, "start": gene.start, "stop": gene.end, "strand": gene.strand, "status": row["status"],
                      **{f"{side}_length": row.get(f"{side}_length", "") for side in ("left", "right")},
                      **{f"{side}_gc": _gc(str(records[side].seq)) if records else "" for side in ("left", "right")}})
        if records: units.extend(search_units(row, records))
        if records: flank_sequences.extend(str(records[side].seq) for side in ("left", "right"))

    output_root = panel_dir(panel) / "proteins"
    matches, vmatch_runs = vmatch.discover_units(units, output_root, show_progress=False)
    evidence = {"inputs": [], "raw_candidates": []}
    arrays, trf_runs = trf.discover_units(units, output_root, run_missing=True, show_progress=False, normalization_evidence=evidence)
    ledger = {"genome": genome, "annotated_cds": len(exclusions), **exclusions.replace("", "eligible").value_counts().to_dict(),
              "drawn": len(genes), "valid": sum(gene["status"] == "valid" for gene in genes),
              "ambiguous_flanks": sum(not set(str(records_side)) <= set("ACGT") for records_side in flank_sequences),
              "left_units": sum(unit["database_side"] == "left" for unit in units),
              "right_units": sum(unit["database_side"] == "right" for unit in units)}

    return {"genome": genome, "ledger": ledger, "genes": genes, "units": [vmatch._unit_output_row(unit, output_root) for unit in units],
            "matches": matches, "vmatch_runs": vmatch_runs, "arrays": arrays, "trf_runs": trf_runs, "evidence": evidence}

def _gc(sequence): return round(sum(base in "GC" for base in sequence) / len(sequence), 4) if sequence else ""

def panel_paths(panel):

    directory = panel_dir(panel)

    return {"contexts_path": directory / "step3c_contexts.tsv", "matches_path": directory / "step3c_matches.tsv",
            "vmatch_runs_path": directory / "step3c_runs.tsv", "trf_arrays_path": directory / "step3c_trf_arrays.tsv",
            "trf_runs_path": directory / "step3c_trf_runs.tsv",
            "trf_normalization_fingerprint_path": directory / "step3c_trf_normalization_fingerprint.json",
            "genes_path": directory / "genes.tsv", "ledger_path": directory / "genome_ledger.tsv",
            "units_path": directory / "background_units.tsv", "focal_check_path": directory / "focal_contexts.tsv"}

def search_panel(panel, processes):

    genomes = panel_genomes(panel)
    count = cfg.BACKGROUND_PANELS[panel]["genes_per_genome"]
    contexts = pd.read_csv(cfg.STEP2_CONTEXTS_PATH, sep="\t", dtype=str, keep_default_na=False)
    contexts["genome"] = contexts["canonical_assembly"].where(contexts["canonical_assembly"] != "", contexts["assembly"])
    contexts = contexts.loc[contexts["genome"].isin(genomes), ["genome", "nucleotide_accession", "start", "stop"]]
    roster_ids = frozenset(roster.sequences())
    tasks = [(panel, genome, count, contexts.loc[contexts["genome"] == genome], roster_ids) for genome in genomes]
    results = []
    started = time.time()

    with Pool(processes) as pool:

        for done, result in enumerate(pool.imap_unordered(_search_genome, tasks, chunksize=1), 1):

            results.append(result)
            if done % 50 == 0 or done == len(tasks): print(f"{done}/{len(tasks)} genomes searched, {time.time() - started:,.0f} s", flush=True)

    results.sort(key=lambda result: result["genome"])
    paths = panel_paths(panel)
    gather = lambda key: [row for result in results for row in result[key]]
    vmatch._write_tsv(paths["contexts_path"], vmatch.UNIT_FIELDS, gather("units"))
    vmatch._write_tsv(paths["matches_path"], vmatch.MATCH_FIELDS, gather("matches"))
    vmatch._write_tsv(paths["vmatch_runs_path"], vmatch.RUN_FIELDS, gather("vmatch_runs"))
    arrays = gather("arrays")
    vmatch._write_tsv(paths["trf_arrays_path"], TRF_ARRAY_FIELDS, arrays)
    vmatch._write_tsv(paths["trf_runs_path"], trf.RUN_FIELDS, gather("trf_runs"))
    evidence = {key: [row for result in results for row in result["evidence"][key]] for key in ("inputs", "raw_candidates")}
    fingerprint = trf._normalization_fingerprint_record(evidence, paths["trf_arrays_path"], len(arrays))
    paths["trf_normalization_fingerprint_path"].write_text(json.dumps(fingerprint, indent=2, sort_keys=True))
    vmatch._write_tsv(paths["genes_path"], GENE_FIELDS, gather("genes"))
    pd.DataFrame([result["ledger"] for result in results]).fillna(0).convert_dtypes().to_csv(paths["ledger_path"], sep="\t", index=False)
    print(f"Searched {len(genomes):,} genomes in {time.time() - started:,.0f} s: {paths['contexts_path'].parent}")

@dataclass(frozen=True, slots=True)
class SearchedFlank:

    protein_accession: str
    scope: str
    search_unit_id: str
    source_context_id: str
    length: int

    @property
    def database_side(self): return self.scope.removeprefix("within_")

    @property
    def representative_unit_id(self): return f"{self.protein_accession}:{self.scope}"

    represented_context_count = 1

    def searched_bases_within(self, cutoff): return min(self.length, cutoff)

def measure_panel(panel):

    paths = panel_paths(panel)
    inputs = step4c.load_inputs(paths)
    suppression = step4c.accepted_build_strategies()["vmatch_suppression"]
    settings = suppression["settings"]
    retained, _ = suppression["partition"](inputs["repeats"], inputs["arrays"], minimum_arm_coverage=float(settings["minimum_arm_coverage"]),
                                           minimum_overlap_bp=int(settings["minimum_overlap_bp"]))
    composition = build_composition_rows(inputs["units"], inputs["representative_sequences"])
    items = build_evidence_items(tuple(retained), inputs["arrays"], inputs["units"], composition)
    flanks = [SearchedFlank(protein, scope, unit_id, row["source_context_id"], int(row[f"{row['database_side']}_flank_length"]))
              for (protein, scope, unit_id), row in inputs["units"].items()]
    rows = []

    for repeat_set in cfg.STEP5_REPEAT_SETS:

        spans = {}

        for item in items:

            if not counts_evidence(repeat_set, item): continue
            key = f"{item.protein_accession}:{item.scope}"
            spans.setdefault(key, []).extend((stretch.start, stretch.end) for stretch in item.stretches)

        for genome, group in pd.Series(flanks).groupby([flank.protein_accession.split("__")[0] for flank in flanks]):

            rows.extend(searched_unit_rows(repeat_set, genome, list(group), spans))

    table = pd.DataFrame(rows).rename(columns={"group_accession": "genome"})
    table.to_csv(paths["units_path"], sep="\t", index=False)
    print(f"{len(items):,} evidence items over {len(flanks):,} flanks: {paths['units_path']}")

    return table

def main():

    parser = argparse.ArgumentParser(description="Stage 05C background panel (T015).")
    parser.add_argument("action", choices=("focal-check", "search", "measure"))
    parser.add_argument("--panel", choices=tuple(cfg.BACKGROUND_PANELS), default="pilot")
    parser.add_argument("--processes", type=int, default=60)
    arguments = parser.parse_args()

    if arguments.action == "focal-check":

        table = focal_check(panel_genomes(arguments.panel))
        panel_dir(arguments.panel).mkdir(parents=True, exist_ok=True)
        table.to_csv(panel_paths(arguments.panel)["focal_check_path"], sep="\t", index=False)
        print(table[["status", "cut_matches", "boundaries_match", "cds_found", "cds_pseudo", "cds_partial"]].astype(str).apply(pd.Series.value_counts).to_string())

    elif arguments.action == "search": search_panel(arguments.panel, arguments.processes)
    else: measure_panel(arguments.panel)

if __name__ == "__main__":

    main()
