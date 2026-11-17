import pandas as pd
from Bio import Phylo, SeqIO

from src import config_main as main_cfg
from src import roster
from src.protein_tree import config_tree as cfg

def select():

    tree = Phylo.read(cfg.PILOT_TREE_PATH, "newick")
    pilot = {record.id: str(record.seq).upper().rstrip("*") for record in SeqIO.parse(cfg.PILOT_FASTA_PATH, "fasta")}
    proteins = pd.read_csv(cfg.PILOT_PROTEINS_PATH, usecols=["Accession", "Description", "Tree", "Feature Protein ID"]).query("Tree == 1")
    proteins = proteins.set_index(proteins["Accession"].str.split(".").str[0], verify_integrity=True)
    sequences = {accession: sequence.upper() for accession, sequence in roster.sequences().items()}
    in_roster = set(sequences.values())
    known = [*in_roster, *pilot.values()]
    anchors = {anchor["Name"]: anchor["protein_acc"] for anchor in main_cfg.the_nine.values()}
    tips = {name: next(genome for genome, sequence in pilot.items() if sequence == sequences[anchors[name]]) for name in cfg.RELATIVE_ANCHORS}
    shortest = min((sequences[anchors[name]] for name in cfg.RELATIVE_ANCHORS), key=len)
    picked = {}

    for name, tip in tips.items():

        candidates = sorted((tree.distance(tip, genome), genome) for genome in pilot if genome != tip)
        chosen = [(distance, genome) for distance, genome in candidates if _whole(pilot[genome], sequences[anchors[name]], known, in_roster)]

        for distance, genome in chosen[:cfg.RELATIVES_PER_ANCHOR]: picked[genome] = ("nearest", name, distance)

    span = tree.common_ancestor(*picked, *tips.values())

    for genome in sorted(tip.name for tip in span.get_terminals()):

        if genome in picked or genome in tips.values() or not _whole(pilot[genome], shortest, known, in_roster): continue
        distance, name = min((tree.distance(tips[name], genome), name) for name in tips)
        picked[genome] = ("between", name, distance)

    rows = [{"role": role, "anchor": name, "protein_accession": proteins.at[genome, "Feature Protein ID"], "genome": genome,
             "description": proteins.at[genome, "Description"], "distance": round(distance, 6),
             "length": len(pilot[genome]), "sequence": pilot[genome]} for genome, (role, name, distance) in picked.items()]
    table = pd.DataFrame(rows).sort_values(["role", "anchor", "distance"], ascending=[False, True, True])

    if table["protein_accession"].duplicated().any() or table["protein_accession"].isin(list(sequences)).any(): raise ValueError("A relative is chosen twice or is already a roster protein")

    cfg.RELATIVES_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(cfg.RELATIVES_PATH, sep="\t", index=False)

    return table

def _whole(sequence, anchor, known, in_roster):

    if sequence in in_roster or not sequence.startswith("M"): return False
    if len(sequence) < cfg.RELATIVE_MIN_LENGTH_SHARE * len(anchor): return False

    return not any(len(other) > len(sequence) and sequence in other for other in known)

def read():

    table = pd.read_csv(cfg.RELATIVES_PATH, sep="\t")

    return dict(zip(table["protein_accession"], table["sequence"]))
