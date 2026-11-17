import math

import pandas as pd

from src.genomes import config_genomes as genomes_cfg
from src.stage02_domains import config_domains as domains_cfg
from src.stage05_C_vmatch import background as panel
from src.stage05_C_vmatch import config_vmatch as vmatch_cfg
from src.stage06_repeat_enrichment import config_enrichment as enrichment_cfg
from src.stage06_repeat_enrichment.background import read_flanks
from src.stage07_figures import config_figures as cfg

SIDES = {"left": "Upstream", "right": "Downstream"}
WINDOWS = vmatch_cfg.STEP5_UNIT_WINDOWS
REPEAT_SETS = {"all": "all", "tandem": "tandem", "direct": "direct", "inverted": "inverted", "all_strict": "all, strict cutoff"}
EXCLUSIONS = {"protein_not_intact": "protein not intact", "genome_not_in_panel": "genome not among the RefSeq background genomes",
              "cds_partial_or_pseudo": "gene annotated as partial or as a pseudogene", "ambiguous_bases": "bases other than A, C, G or T"}

def searched(window): return f"Searched nt within {window} nt"

def covered(window, name): return f"Repeat nt within {window} nt: {name}"

def exponent(value): return f"10^{round(math.log10(value))}"

MEASURES = [column for window in WINDOWS for column in (searched(window), *(covered(window, name) for name in REPEAT_SETS.values()))]
GENE = ["Protein accession", "Assembly", "Genus", "Contig", "Gene start", "Gene end", "Strand", "Side", "Flank GC"]
HOMOLOG_SHOWN = [*GENE[:1], "HTH class", "Cluster", *GENE[1:], "Contexts represented", "Left out because", *MEASURES]
BACKGROUND_SHOWN = [*GENE, "Left out because", *MEASURES]
NO_HOMOLOG = "genome holds no homolog flank that entered the comparison"

NOTES = {
    "Protein accession": "NCBI RefSeq protein accession of the gene whose flank this row describes.",
    "HTH class": "Homolog flanks only. no_HTH: no helix-turn-helix (HTH) domain. HTH+: one. HTH++: two or more.",
    "Cluster": "Homolog flanks only. The homolog's MMseqs2 cluster, as in Table S4.",
    "Assembly": "NCBI assembly accession of the genome the flank was cut from.",
    "Genus": "Genus of that genome's organism.",
    "Contig": "Nucleotide accession of the sequence that carries the gene.",
    "Gene start": "First base of the gene on the contig, 1-based.",
    "Gene end": "Last base of the gene on the contig, 1-based.",
    "Strand": "Strand of the gene on the contig.",
    "Side": "Upstream: the flank ending at the start codon. Downstream: the flank beginning at the stop codon.",
    "Flank GC": "GC fraction of the flank's searched sequence, up to 1,000 nt from the gene.",
    "Contexts represented": f"Homolog flanks only. Number of same-side flanks of this homolog, from different genomes, that differ by at most "
                            f"{1 - vmatch_cfg.FLANK_CLUSTER_MIN_PREFIX_SIMILARITY:.0%} edits over their gene-proximal part and are "
                            f"represented by this one flank.",
    "Left out because": "Empty for flanks that entered the comparison of homolog and background flanks; otherwise the reason the flank "
                        "was left out. A background flank is compared with the homolog flanks of its own genome.",
    searched("W"): "Bases of the flank searched within W nt of the gene. A flank enters a comparison at W only when this equals W.",
    covered("W", "<repeat set>"): "Bases within W nt of the gene covered by repeats of that set, overlaps counted once. A flank counts as carrying a repeat at W when this is above 0.",
    "Repeat sets": f"tandem: tandem arrays called by Tandem Repeats Finder. direct: direct repeat pairs from Vmatch whose copies lie at "
                   f"most {vmatch_cfg.VMATCH_MAX_SPACER} nt apart. inverted: inverted repeat pairs from Vmatch. all: the three together. "
                   f"Pairs are kept at an E-value (first-order Markov model) of {exponent(vmatch_cfg.STEP5_MAX_PAIR_EVALUE)} or less. "
                   f"all, strict cutoff: the three together, with pairs kept at {exponent(vmatch_cfg.STEP5_STRICT_MAX_PAIR_EVALUE)} or less.",
    "Windows": f"W is {', '.join(map(str, WINDOWS[:-1]))} or {WINDOWS[-1]} nt."}

def wide(flanks, keys):

    measures = {f"repeat_bases_{window}": covered for window in WINDOWS}
    pivoted = flanks.pivot(index=keys, columns="repeat_set", values=list(measures))
    pivoted.columns = [measures[field](field.rsplit("_", 1)[1], REPEAT_SETS[name]) for field, name in pivoted.columns]
    bases = flanks.drop_duplicates(keys).set_index(keys)[[f"searched_bases_{window}" for window in WINDOWS]]
    bases.columns = [searched(window) for window in WINDOWS]

    return bases.join(pivoted, validate="one_to_one").reset_index()

def homolog_flanks():

    flanks = read_flanks(enrichment_cfg.BACKGROUND_PANEL)
    keys = ["protein_accession", "source_context_id", "side"]
    result = wide(flanks, keys)
    described = flanks.drop_duplicates(keys)[[*keys, "hth_class", "genome", "genus", "gc", "represented_contexts", "exclusion"]]
    result = described.merge(result, on=keys, validate="one_to_one")
    places = panel.focal_units()[["protein_accession", "source_context_id", "nucleotide_accession", "start", "stop", "strand"]].drop_duplicates()
    result = result.merge(places, on=["protein_accession", "source_context_id"], how="left", validate="many_to_one")
    clusters = pd.read_csv(domains_cfg.DPAM_PROTEIN_ARCHITECTURES_PATH, usecols=["protein_accession", "cluster_id"])
    result = result.merge(clusters, on="protein_accession", how="left", validate="many_to_one")
    result["exclusion"] = result["exclusion"].map(lambda code: EXCLUSIONS[code] if code else "")
    names = {"protein_accession": "Protein accession", "hth_class": "HTH class", "cluster_id": "Cluster", "genome": "Assembly", "genus": "Genus",
             "nucleotide_accession": "Contig", "start": "Gene start", "stop": "Gene end", "strand": "Strand", "side": "Side", "gc": "Flank GC",
             "represented_contexts": "Contexts represented", "exclusion": "Left out because"}
    result = result.rename(columns=names)
    result["Side"] = result["Side"].map(SIDES)

    return result.sort_values(["Protein accession", "Assembly", "Side"], ascending=[True, True, False])[HOMOLOG_SHOWN]

def background_flanks(homologs):

    paths = panel.panel_paths(enrichment_cfg.BACKGROUND_PANEL)
    keys = ["protein_accession", "side"]
    result = wide(pd.read_csv(paths["units_path"], sep="\t"), keys)
    genes = pd.read_csv(paths["genes_path"], sep="\t")
    gc = genes.melt(id_vars="sequence_id", value_vars=["left_gc", "right_gc"], var_name="side", value_name="gc")
    gc["side"] = gc["side"].str.removesuffix("_gc")
    result = result.merge(gc, left_on=keys, right_on=["sequence_id", "side"], how="left", validate="one_to_one").drop(columns="sequence_id")
    places = genes[["sequence_id", "genome", "protein_id", "seqid", "start", "stop", "strand"]]
    result = result.merge(places, left_on="protein_accession", right_on="sequence_id", how="left", validate="many_to_one")
    organisms = pd.read_csv(genomes_cfg.MANIFEST_PATH, sep="\t", usecols=["accession", "organism"]).set_index("accession")["organism"]
    result["genus"] = result["genome"].map(organisms).map(panel._genus)
    names = {"protein_id": "Protein accession", "genome": "Assembly", "genus": "Genus", "seqid": "Contig", "start": "Gene start", "stop": "Gene end",
             "strand": "Strand", "side": "Side", "gc": "Flank GC"}
    result = result.drop(columns="protein_accession").rename(columns=names)
    result["Side"] = result["Side"].map(SIDES)
    compared = set(homologs.loc[homologs["Left out because"] == "", "Assembly"])
    result["Left out because"] = result["Assembly"].map(lambda genome: "" if genome in compared else NO_HOMOLOG)

    return result.sort_values(["Assembly", "Protein accession", "Side"], ascending=[True, True, False])[BACKGROUND_SHOWN]

def draw():

    homologs = homolog_flanks()
    sheets = {"Homolog flanks": homologs, "Background flanks": background_flanks(homologs)}
    paths = {name: cfg.FLANK_TABLE_DIR / f"table_S5_{name.split()[0].lower()}_flanks.csv" for name in sheets}
    excel_path = cfg.FLANK_TABLE_DIR / "table_S5_flanks.xlsx"
    notes = pd.DataFrame({"Column": list(NOTES), "Description": list(NOTES.values())})

    cfg.FLANK_TABLE_DIR.mkdir(parents=True, exist_ok=True)

    for name, table in sheets.items(): table.to_csv(paths[name], index=False)

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        notes.to_excel(writer, sheet_name="Columns", index=False)
        writer.sheets["Columns"].column_dimensions["A"].width = 40
        writer.sheets["Columns"].column_dimensions["B"].width = 120

        for name, table in sheets.items():

            table.to_excel(writer, sheet_name=name, index=False)
            writer.sheets[name].freeze_panes = "B2"

    print(f"Table S5: {len(homologs):,} homolog flanks, {(homologs['Left out because'] == '').sum():,} used; "
          f"{len(sheets['Background flanks']):,} background flanks, {(sheets['Background flanks']['Left out because'] == '').sum():,} used")

    return [*paths.values(), excel_path]
