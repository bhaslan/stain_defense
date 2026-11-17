from concurrent.futures import ProcessPoolExecutor
from itertools import product

import pandas as pd

from src.genomes.assembly import Assembly
from src.stage05_C_vmatch import background as panel
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage06_repeat_enrichment import background
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage07_figures import config_figures as cfg

PARTS = ("upstream", "gene", "downstream", "context", "gc3")
CONTEXTS_NAME = "context_gc.tsv"

def _gc(sequence): return sum(base in "GC" for base in sequence) / len(sequence)

def _cut(task):

    genome, rows = task
    assembly = Assembly(genome)
    reach = vmatch_cfg.FLANK_LENGTH
    measured = []

    for row in rows:

        name = assembly.names.get(row["sequence_name"])

        if name is None:

            measured.append({**row, "set_aside": "sequence absent"})
            continue

        sequence, start, stop = assembly.sequences[name], row["start"], row["stop"]

        if start - reach < 1 or stop + reach > len(sequence):

            measured.append({**row, "set_aside": "flank short"})
            continue

        before, gene, after = sequence[start - 1 - reach:start - 1], sequence[start - 1:stop], sequence[stop:stop + reach]

        if not set(before + gene + after) <= set("ACGT"):

            measured.append({**row, "set_aside": "ambiguous bases"})
            continue

        if row["strand"] == "-": before, gene, after = after, gene[::-1], before

        measured.append({**row, "set_aside": "", "gene_length": len(gene), "upstream": _gc(before), "gene": _gc(gene), "downstream": _gc(after),
                         "context": _gc(before + gene + after), "gc3": _gc(gene[2::3])})

    return measured

def contexts():

    flanks = background.read_flanks()
    used = flanks.loc[(flanks["exclusion"] == "") & (flanks["repeat_set"] == vmatch_cfg.STEP5_PRIMARY_REPEAT_SET)]
    used = used.drop_duplicates(["protein_accession", "source_context_id"])[["group_accession", "protein_accession", "source_context_id", "genome", "genus", "hth_class"]]
    step2 = pd.read_csv(vmatch_cfg.STEP2_CONTEXTS_PATH, sep="\t", dtype=str, keep_default_na=False, usecols=["protein_accession", "context_id", "nucleotide_accession", "start", "stop", "strand"])
    focal = used.merge(step2, left_on=["protein_accession", "source_context_id"], right_on=["protein_accession", "context_id"], validate="one_to_one").drop(columns="context_id")
    focal = focal.rename(columns={"nucleotide_accession": "sequence_name"}).assign(kind="focal")
    genes = pd.read_csv(panel.panel_paths(enrichment_cfg.BACKGROUND_PANEL)["genes_path"], sep="\t")
    whole = (genes["status"] == "valid") & (genes["left_length"] == vmatch_cfg.FLANK_LENGTH) & (genes["right_length"] == vmatch_cfg.FLANK_LENGTH)
    ordinary = genes.loc[whole & genes["genome"].isin(focal["genome"]), ["genome", "sequence_id", "seqid", "start", "stop", "strand"]]
    ordinary = ordinary.rename(columns={"seqid": "sequence_name"}).assign(kind="ordinary")
    rows = pd.concat([focal, ordinary], ignore_index=True).astype({"start": int, "stop": int})
    tasks = [(genome, group.to_dict("records")) for genome, group in rows.groupby("genome")]

    with ProcessPoolExecutor(cfg.CONTEXT_GC_WORKERS) as pool: table = pd.DataFrame([row for measured in pool.map(_cut, tasks) for row in measured])

    print("Contexts set aside:", table.groupby(["kind", "set_aside"]).size().to_dict())
    cfg.REPEAT_CHARACTER_BUILD_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.REPEAT_CHARACTER_BUILD_DIR / CONTEXTS_NAME, sep="\t", index=False)

    return table.loc[table["set_aside"] == ""]

def offsets(table):

    ordinary = table.loc[table["kind"] == "ordinary"].groupby("genome")[list(PARTS)].mean()
    kept = table.loc[table["genome"].isin(ordinary.index)]

    return kept.assign(**{part: 100 * (kept[part] - ordinary.loc[kept["genome"], part].to_numpy()) for part in PARTS})

def compare(table, path):

    ordinary = table.loc[table["kind"] == "ordinary"].groupby("genome")[list(PARTS)].mean()
    focal = table.loc[(table["kind"] == "focal") & table["genome"].isin(ordinary.index)]
    rows = []

    for (name, members), part in product(background.CLASSES.items(), PARTS):

        chosen = focal.loc[focal["hth_class"].isin(members)]
        if chosen.empty: continue
        rows.append({"hth_class": name, "part": part, "samples": chosen["group_accession"].nunique(), "contexts": len(chosen), "genera": chosen["genus"].nunique(),
                     **background._offset(chosen.assign(gc=chosen[part]), ordinary.loc[chosen["genome"], part].to_numpy())})

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)
